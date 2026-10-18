"""GPU rendering for the chain cut: one switch, `--renderer=opengl`.

With Cairo (the default) this module exports Manim's own classes and
patches nothing, so a Cairo render stays byte for byte the same.

With `--renderer=opengl` (the CLI sets config.renderer before it imports
the scene file) it makes Manim's OpenGL renderer draw what Cairo draws:

- Patches on import: OpenGLMobject gets Cairo's `z_index` and
  `set_z_index`, and `family=` is accepted where Cairo takes it
  (set_fill, set_stroke, set_color, set_opacity).
- Exports `ImageMobject` = GLImage, `VMobject` = OpenGLVMobject and
  `Mobject` = OpenGLMobject, so the scenes' isinstance checks keep their
  meaning. Import those three from here, not from manim.
- GLRenderer, which `swap_renderer` puts in place of the CLI's renderer
  (style.BeatScene does it). It draws the flattened family in Cairo's
  order (a stable sort by z_index), per member back stroke, fill, stroke;
  it places and resizes every image exactly as Cairo's camera does (same
  integer pixel box, same PIL filter), so a still image lands on the same
  pixels; it keeps Cairo's clock (`time` counts written frames, so hooks
  that read it see the same times); it renders headless into an MSAA
  framebuffer.

Anti-aliasing. Manim's fill shader feathers every edge outwards by its
anti-alias width (1.5 px), so shapes and text come out about half a pixel
bolder than Cairo's, whose coverage is centred on the edge. Here fills get
(almost) no shader band and are shaded per MSAA sample instead, so curved
and straight edges alike get the samples' coverage, centred like Cairo's.
Strokes keep the stroke shader's own centred feather.

Knobs (environment): GL_SAMPLES (MSAA samples, default 8; 0 = none),
GL_FILL_AA_PX (fill feather in pixels, default 0 = per-sample MSAA only),
GL_STROKE_AA_PX (stroke feather in pixels, default 1.5), GL_STATS=1
(per-frame timings at the end).
"""
from __future__ import annotations

import functools
import os
import time
import weakref
from pathlib import Path, PurePath

import numpy as np
from manim import config
from manim.constants import RendererType

GL = config.renderer == RendererType.OPENGL

if not GL:
    from manim import ImageMobject, Mobject, VMobject  # noqa: F401  (the stock classes, unchanged)

    def swap_renderer(renderer):
        return renderer

else:
    import moderngl
    from manim import DOWN, LEFT, RIGHT, UP
    from manim.constants import DEFAULT_QUALITY, QUALITIES
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject
    from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
    from manim.renderer.opengl_renderer import OpenGLRenderer
    from manim.renderer.shader_wrapper import get_shader_code_from_file
    from manim.scene.scene_file_writer import SceneFileWriter
    from manim.utils.color import color_to_rgba
    from manim.utils.exceptions import EndSceneEarlyException
    from manim.utils.images import change_to_rgba_array, get_full_raster_image_path
    from manim.utils.iterables import list_update
    from manim.utils.space_ops import angle_of_vector
    from PIL import Image

    Mobject = OpenGLMobject
    VMobject = OpenGLVMobject

    # ---- patches: Cairo's z_index and family= on the OpenGL classes ----------

    def _set_z_index(self, z_index_value: float, family: bool = True):
        """Cairo's Mobject.set_z_index: the submobjects first, then self."""
        if family:
            for sm in self.submobjects:
                sm.set_z_index(z_index_value, family=family)
        self.z_index = z_index_value
        return self

    def _family_alias(fn):
        """Accept Cairo's `family=` as OpenGL's `recurse=`."""
        @functools.wraps(fn)
        def wrapper(self, *args, family=None, **kwargs):
            if family is not None:
                kwargs.setdefault("recurse", family)
            return fn(self, *args, **kwargs)
        wrapper._family_alias = True
        return wrapper

    if not getattr(OpenGLMobject.set_opacity, "_family_alias", False):
        OpenGLMobject.z_index = 0
        OpenGLMobject.set_z_index = _set_z_index
        for cls, names in ((OpenGLMobject, ("set_opacity", "set_color")),
                           (OpenGLVMobject, ("set_fill", "set_stroke", "set_opacity", "set_color"))):
            for name in names:
                if name in cls.__dict__:
                    setattr(cls, name, _family_alias(cls.__dict__[name]))

    # ---- images --------------------------------------------------------------

    class GLImage(OpenGLMobject):
        """Cairo's ImageMobject as an OpenGL mobject: four corner points (UL,
        UR, DL, DR), a `pixel_array` the scenes swap, a resampling algorithm,
        and its opacity in `rgbas` (so FadeIn/FadeOut interpolate it). The
        renderer resizes the array with PIL to the pixel box Cairo would
        use and draws it 1:1."""

        def __init__(self, filename_or_array, scale_to_resolution: int = QUALITIES[DEFAULT_QUALITY]["pixel_height"],
                     invert: bool = False, image_mode: str = "RGBA", resampling_algorithm: int = Image.BICUBIC,
                     **kwargs):
            if isinstance(filename_or_array, (str, PurePath)):
                path = get_full_raster_image_path(filename_or_array)
                arr = np.array(Image.open(path).convert(image_mode))
                self.path = path
            else:
                arr = np.array(filename_or_array)
            arr = change_to_rgba_array(arr, "uint8")
            if invert:
                arr[:, :, :3] = 255 - arr[:, :, :3]
            self._version = 0
            self.pixel_array = arr
            self.orig_alpha_pixel_array = arr[:, :, 3].copy()
            self.scale_to_resolution = scale_to_resolution
            self.set_resampling_algorithm(resampling_algorithm)
            super().__init__(**kwargs)

        @property
        def pixel_array(self) -> np.ndarray:
            return self._pixels

        @pixel_array.setter
        def pixel_array(self, arr: np.ndarray) -> None:
            self._pixels = arr
            self._version += 1

        # Cairo bakes a fade into the alpha channel; here the opacity is drawn
        # by the shader, so the scenes' "keep a fade in progress" branches
        # (`if img.stroke_opacity < 1`) must not fire.
        @property
        def stroke_opacity(self) -> float:
            return 1.0

        @stroke_opacity.setter
        def stroke_opacity(self, value) -> None:
            pass

        @property
        def fill_opacity(self) -> float:
            return 1.0

        @fill_opacity.setter
        def fill_opacity(self, value) -> None:
            pass

        def init_points(self) -> None:
            self.set_points(np.array([UP + LEFT, UP + RIGHT, DOWN + LEFT, DOWN + RIGHT], dtype=float))
            self.center()
            h, w = self.pixel_array.shape[:2]
            height = h / self.scale_to_resolution * config["frame_height"] if self.scale_to_resolution else 3
            self.stretch_to_fit_height(height)
            self.stretch_to_fit_width(height * w / h)

        def get_pixel_array(self) -> np.ndarray:
            return self.pixel_array

        def set_resampling_algorithm(self, resampling_algorithm: int):
            if not isinstance(resampling_algorithm, int):
                raise ValueError("resampling_algorithm has to be an int (a PIL filter constant)")
            self.resampling_algorithm = resampling_algorithm
            return self

        def get_shader_wrapper_list(self):
            return []

    ImageMobject = GLImage

    # ---- the renderer ----------------------------------------------------------

    _IMAGE_VS = """
    #version 330
    in vec2 in_pos;
    in vec2 in_uv;
    out vec2 v_uv;
    void main() { gl_Position = vec4(in_pos, 0.0, 1.0); v_uv = in_uv; }
    """
    _IMAGE_FS = """
    #version 330
    uniform sampler2D tex;
    uniform float opacity;
    in vec2 v_uv;
    out vec4 frag_color;
    void main() { vec4 c = texture(tex, v_uv); frag_color = vec4(c.rgb, c.a * opacity); }
    """

    def _program(ctx, folder: str, per_sample: bool = False):
        """A program from Manim's shader folder; per_sample: the fragment
        shader runs once per MSAA sample (GLSL 4.0 `sample` input)."""
        def code(name):
            return get_shader_code_from_file(Path(folder) / f"{name}.glsl")
        geom, frag = code("geom"), code("frag")
        if per_sample:      # both sides of the varying need the qualifier
            for src, decl in ((geom, "out vec2 uv_coords;"), (frag, "in vec2 uv_coords;")):
                if decl not in src:
                    raise RuntimeError(f"{folder}: no `{decl}` to qualify")
            geom = geom.replace("#version 330", "#version 400", 1).replace(
                "out vec2 uv_coords;", "sample out vec2 uv_coords;", 1)
            frag = frag.replace("#version 330", "#version 400", 1).replace(
                "in vec2 uv_coords;", "sample in vec2 uv_coords;", 1)
        return ctx.program(vertex_shader=code("vert"), geometry_shader=geom, fragment_shader=frag)

    def _vao_format(prog, layout) -> tuple[str, list[str]]:
        """moderngl format for a packed float32 record, skipping the fields
        the program does not use."""
        parts, names = [], []
        for name, n in layout:
            try:
                prog[name]
                parts.append(f"{n}f")
                names.append(name)
            except KeyError:
                parts.append(f"{4 * n}x")
        return " ".join(parts), names

    class _Buf:
        """A growable GPU buffer, refilled once per frame. `gen` counts
        reallocations, so vertex arrays that read it know to rebuild."""

        def __init__(self, ctx, reserve: int = 1 << 20):
            self.ctx = ctx
            self.buf = ctx.buffer(reserve=reserve, dynamic=True)
            self.gen = 0

        def write(self, data: bytes) -> None:
            if len(data) > self.buf.size:
                self.buf.release()
                self.buf = self.ctx.buffer(reserve=max(len(data), 2 * self.buf.size), dynamic=True)
                self.gen += 1
            if data:
                self.buf.orphan(self.buf.size)
                self.buf.write(data)

    class GLRenderer(OpenGLRenderer):
        """Manim's OpenGL renderer with Cairo's draw order, clock and image
        placement. Headless; draws into an MSAA framebuffer, resolves, reads."""

        def __init__(self, file_writer_class=SceneFileWriter, skip_animations: bool = False):
            super().__init__(file_writer_class=file_writer_class, skip_animations=skip_animations)
            self.time = 0.0
            self.static_image = None
            self.samples = int(os.environ.get("GL_SAMPLES", "8"))
            self.fill_aa_px = float(os.environ.get("GL_FILL_AA_PX", "0"))
            self.stroke_aa_px = float(os.environ.get("GL_STROKE_AA_PX", "1.5"))
            self._tex: dict[int, list] = {}          # id(image) -> [weakref, key, texture, last frame]
            self._bg_src = None
            self._bg = None
            self.stats = {"frames": 0, "written": 0, "build": 0.0, "draw": 0.0, "read": 0.0, "draws": 0,
                          "leaves": 0, "img_uploads": 0}

        # ---- GL setup
        def init_scene(self, scene) -> None:
            self.partial_movie_files = []
            self.file_writer = self._file_writer_class(self, scene.__class__.__name__)
            self.scene = scene
            try:
                ctx = moderngl.create_context(standalone=True)
            except Exception:
                ctx = moderngl.create_context(standalone=True, backend="egl")
            self.context = ctx
            W, H = config["pixel_width"], config["pixel_height"]
            self.fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 4)])
            if self.samples > 0:
                self.fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((W, H), 4, samples=self.samples)])
            else:
                self.fbo_ms = self.fbo
            self.frame_buffer_object = self.fbo
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE)

            self.p_fill = _program(ctx, OpenGLVMobject.fill_shader_folder,
                                   per_sample=self.samples > 0 and self.fill_aa_px == 0)
            self.p_stroke = _program(ctx, OpenGLVMobject.stroke_shader_folder)
            self.p_image = ctx.program(vertex_shader=_IMAGE_VS, fragment_shader=_IMAGE_FS)
            self.p_image["tex"].value = 0
            self._fmt = {
                "fill": _vao_format(self.p_fill, [("point", 3), ("unit_normal", 3), ("color", 4), ("vert_index", 1)]),
                "stroke": _vao_format(self.p_stroke, [("point", 3), ("prev_point", 3), ("next_point", 3),
                                                      ("unit_normal", 3), ("stroke_width", 1), ("color", 4)]),
                "image": ("2f 2f", ["in_pos", "in_uv"]),
            }
            self._bufs = {k: _Buf(ctx) for k in ("fill", "fill_idx", "stroke", "image")}
            self._vaos: dict[str, tuple] = {}
            info = ctx.info
            print(f"[glcompat] GL_RENDERER={info.get('GL_RENDERER')} GL_VERSION={info.get('GL_VERSION')} "
                  f"samples={self.samples} fill_aa_px={self.fill_aa_px} stroke_aa_px={self.stroke_aa_px} "
                  f"frame={W}x{H}", flush=True)

        def _vao(self, kind: str):
            """The vertex array for a stream, rebuilt after its buffers grew."""
            bufs = [self._bufs[kind]] + ([self._bufs["fill_idx"]] if kind == "fill" else [])
            gens = tuple(b.gen for b in bufs)
            have = self._vaos.get(kind)
            if have is None or have[0] != gens:
                if have is not None:
                    have[1].release()
                prog = {"fill": self.p_fill, "stroke": self.p_stroke, "image": self.p_image}[kind]
                fmt, names = self._fmt[kind]
                kw = {"index_buffer": bufs[1].buf, "index_element_size": 4} if kind == "fill" else {}
                vao = self.context.vertex_array(prog, [(bufs[0].buf, fmt, *names)], **kw)
                self._vaos[kind] = have = (gens, vao)
            return have[1]

        def _background(self):
            src = getattr(self.camera, "background_color", None) or config["background_color"]
            if src is not self._bg_src:
                self._bg_src = src
                self._bg = tuple(color_to_rgba(src, 1.0))
            return self._bg

        # ---- the frame
        def _leaves(self, scene) -> list:
            """Cairo's list: the family members with points of every scene
            mobject (the last occurrence kept), stably sorted by z_index."""
            flat = []
            for m in list_update(scene.mobjects, scene.foreground_mobjects):
                flat.extend(f for f in m.get_family() if f.has_points())
            seen, out = set(), []
            for f in reversed(flat):
                if id(f) not in seen:
                    seen.add(id(f))
                    out.append(f)
            out.reverse()
            out.sort(key=lambda f: f.z_index)
            return out

        def _image_quad(self, img, fw: float, fh: float, W: int, H: int):
            """Cairo's display_image_mobject, up to the paste: the integer
            pixel box and the resized (maybe rotated) image. Returns (texture,
            x0, y0, w, h) with the texture current."""
            pts = img.points[:4]
            px = np.empty((4, 2))
            px[:, 0] = pts[:, 0] * (W / fw) + W / 2
            px[:, 1] = pts[:, 1] * (-(H / fh)) + H / 2
            ul, ur, dl = (c.astype(int) for c in px.astype(int)[:3])
            right, down = ur - ul, dl - ul
            centre = ul + (right + down) / 2
            pw = max(int(np.sqrt(float(right @ right))), 1)
            ph = max(int(np.sqrt(float(down @ down))), 1)
            angle = -int(360 * angle_of_vector(right) / (2 * np.pi))
            key = (img._version, pw, ph, img.resampling_algorithm, angle)
            ent = self._tex.get(id(img))
            if ent is None or ent[0]() is not img:
                if ent is not None:
                    ent[2].release()
                ent = self._tex[id(img)] = [weakref.ref(img), None, None, 0]
            if ent[1] != key:
                sub = Image.fromarray(img.pixel_array, mode="RGBA").resize((pw, ph), resample=img.resampling_algorithm)
                if angle != 0:
                    sub = sub.rotate(angle, resample=img.resampling_algorithm, expand=1)
                if ent[2] is not None and ent[2].size == sub.size:
                    ent[2].write(sub.tobytes())
                else:
                    if ent[2] is not None:
                        ent[2].release()
                    tex = self.context.texture(sub.size, 4, data=sub.tobytes())
                    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                    tex.repeat_x = tex.repeat_y = False
                    ent[2] = tex
                ent[1] = key
                self.stats["img_uploads"] += 1
            ent[3] = self.stats["frames"]
            sw, sh = ent[2].size
            x0, y0 = (centre - np.array([sw, sh]) / 2).astype(int)
            return ent[2], int(x0), int(y0), sw, sh

        def update_frame(self, scene) -> None:
            t0 = time.perf_counter()
            self.refresh_perspective_uniforms(scene.camera)
            fw, fh = scene.camera.get_shape()
            W, H = config["pixel_width"], config["pixel_height"]
            fills, idxs, strokes, quads = [], [], [], []
            ops: list[list] = []        # [kind, key, first, count] or ["image", texture, opacity, quad index]
            nfv = nfi = nsv = 0

            def push(kind, key, first, count):
                last = ops[-1] if ops else None
                if last is not None and last[0] == kind and last[1] == key and last[2] + last[3] == first:
                    last[3] += count
                else:
                    ops.append([kind, key, first, count])

            leaves = self._leaves(scene)
            for m in leaves:
                if isinstance(m, GLImage):
                    op = m.get_opacity()
                    if op <= 0:
                        continue
                    tex, x0, y0, sw, sh = self._image_quad(m, fw, fh, W, H)
                    X0, X1 = 2 * x0 / W - 1, 2 * (x0 + sw) / W - 1
                    Y0, Y1 = 1 - 2 * y0 / H, 1 - 2 * (y0 + sh) / H
                    quads.append(np.array([[X0, Y0, 0, 0], [X1, Y0, 1, 0], [X0, Y1, 0, 1], [X1, Y1, 1, 1]], "f4"))
                    ops.append(["image", tex, float(op), len(quads) - 1])
                    continue
                if not isinstance(m, OpenGLVMobject):
                    continue
                fkey = (m.is_fixed_in_frame, m.is_fixed_orientation, tuple(m.fixed_orientation_center), m.gloss,
                        m.shadow)
                stroke = m.has_stroke()
                skey = fkey + (m.joint_type.value, float(m.flat_stroke)) if stroke else None
                if stroke and m.draw_stroke_behind_fill:
                    d = m.get_stroke_shader_data()
                    strokes.append(d)
                    push("stroke", skey, nsv, len(d))
                    nsv += len(d)
                if m.has_fill():
                    d = m.get_fill_shader_data()
                    tri = m.get_triangulation()
                    if len(tri):
                        fills.append(d)
                        idxs.append(np.asarray(tri, "i4") + nfv)
                        push("fill", fkey, nfi, len(tri))
                        nfv += len(d)
                        nfi += len(tri)
                if stroke and not m.draw_stroke_behind_fill:
                    d = m.get_stroke_shader_data()
                    strokes.append(d)
                    push("stroke", skey, nsv, len(d))
                    nsv += len(d)
            # one upload per stream; each draw is a range in it
            if fills:
                self._bufs["fill_idx"].write(np.concatenate(idxs).astype("i4").tobytes())
                self._bufs["fill"].write(np.concatenate(fills).tobytes())
            if strokes:
                self._bufs["stroke"].write(np.concatenate(strokes).tobytes())
            if quads:
                self._bufs["image"].write(np.concatenate(quads).tobytes())
            t1 = time.perf_counter()

            self.fbo_ms.use()
            self.fbo_ms.clear(*self._background())
            pu = dict(self.perspective_uniforms)
            unit = fh / H                   # frame units per pixel
            for prog, aa in ((self.p_fill, max(self.fill_aa_px, 1e-3)), (self.p_stroke, self.stroke_aa_px)):
                pu["anti_alias_width"] = aa * unit
                for name, value in pu.items():
                    try:
                        prog[name].value = value
                    except KeyError:
                        pass
            current = {"fill": None, "stroke": None}
            vaos = {k: self._vao(k) for k, used in (("fill", fills), ("stroke", strokes), ("image", quads)) if used}
            for op in ops:
                kind = op[0]
                if kind == "image":
                    _, tex, opacity, qi = op
                    tex.use(0)
                    self.p_image["opacity"].value = opacity
                    vaos["image"].render(moderngl.TRIANGLE_STRIP, vertices=4, first=4 * qi)
                    continue
                prog = self.p_fill if kind == "fill" else self.p_stroke
                if current[kind] != op[1]:
                    current[kind] = op[1]
                    fixed, orient, centre, gloss, shadow = op[1][:5]
                    vals = {"is_fixed_in_frame": fixed, "is_fixed_orientation": orient,
                            "fixed_orientation_center": centre, "gloss": gloss, "shadow": shadow}
                    if kind == "stroke":
                        vals.update(joint_type=op[1][5], flat_stroke=op[1][6])
                    for name, value in vals.items():
                        try:
                            prog[name].value = value
                        except KeyError:
                            pass
                vaos[kind].render(moderngl.TRIANGLES, vertices=op[3], first=op[2])
            if self.fbo_ms is not self.fbo:
                self.context.copy_framebuffer(self.fbo, self.fbo_ms)
            t2 = time.perf_counter()
            st = self.stats
            st["frames"] += 1
            st["build"] += t1 - t0
            st["draw"] += t2 - t1
            st["draws"] += len(ops)
            st["leaves"] += len(leaves)
            if st["frames"] % 300 == 0:
                self._evict()

        def _evict(self) -> None:
            """Drop textures of images that are gone or unseen for 10 s of frames."""
            now = self.stats["frames"]
            for k, ent in list(self._tex.items()):
                if ent[0]() is None or now - ent[3] > 300:
                    if ent[2] is not None:
                        ent[2].release()
                    del self._tex[k]

        def get_frame(self) -> np.ndarray:
            t0 = time.perf_counter()
            W, H = config["pixel_width"], config["pixel_height"]
            raw = self.fbo.read(components=4, dtype="f1")
            frame = np.flipud(np.frombuffer(raw, np.uint8).reshape(H, W, 4))
            self.stats["read"] += time.perf_counter() - t0
            return frame

        def get_image(self):
            return Image.fromarray(np.ascontiguousarray(self.get_frame()), "RGBA")

        # ---- Cairo's clock and play loop
        def update_skipping_status(self) -> None:
            if self.file_writer.sections[-1].skip_animations:
                self.skip_animations = True
            if config["save_last_frame"]:
                self.skip_animations = True
            if config.from_animation_number > 0 and self.num_plays < config.from_animation_number:
                self.skip_animations = True
            if config.upto_animation_number >= 0 and self.num_plays > config.upto_animation_number:
                self.skip_animations = True
                raise EndSceneEarlyException()

        def play(self, scene, *args, **kwargs) -> None:
            self.skip_animations = self._original_skipping_status
            self.update_skipping_status()
            scene.compile_animation_data(*args, **kwargs)
            if self.skip_animations:
                hash_play = None
                self.time += scene.duration
            else:
                hash_play = f"uncached_{self.num_plays:05}"
            self.file_writer.add_partial_movie_file(hash_play)
            self.animations_hashes.append(hash_play)
            self.file_writer.begin_animation(not self.skip_animations)
            scene.begin_animations()
            if scene.is_current_animation_frozen_frame():
                self.update_frame(scene)
                self.freeze_current_frame(scene.duration)
            else:
                scene.play_internal()
            self.file_writer.end_animation(not self.skip_animations)
            self.num_plays += 1

        def add_frame(self, frame: np.ndarray, num_frames: int = 1) -> None:
            dt = 1 / config["frame_rate"]
            if self.skip_animations:
                return
            self.time += num_frames * dt
            self.stats["written"] += num_frames
            self.file_writer.write_frame(frame, num_frames=num_frames)

        def freeze_current_frame(self, duration: float) -> None:
            dt = 1 / config["frame_rate"]
            self.add_frame(self.get_frame(), num_frames=int(duration / dt))

        def render(self, scene, frame_offset, moving_mobjects) -> None:
            if self.skip_animations:
                return
            self.update_frame(scene)
            self.add_frame(self.get_frame())

        def scene_finished(self, scene) -> None:
            if self.num_plays:
                self.file_writer.finish()
            elif config.write_to_movie:
                config.save_last_frame = True
                config.write_to_movie = False
            if config["save_last_frame"]:
                self.update_frame(scene)
                self.file_writer.save_image(self.get_image())
            if os.environ.get("GL_STATS"):
                st, n = self.stats, max(self.stats["frames"], 1)
                print(f"[glcompat] frames drawn {st['frames']}, written {st['written']}; per drawn frame: "
                      f"build {1e3 * st['build'] / n:.1f} ms, draw {1e3 * st['draw'] / n:.1f} ms, "
                      f"read {1e3 * st['read'] / n:.1f} ms, {st['leaves'] / n:.0f} leaves, "
                      f"{st['draws'] / n:.0f} draw calls; image uploads {st['img_uploads']}", flush=True)

        def clear_screen(self) -> None:
            self.fbo_ms.clear(*self._background())

    # ---- the NVENC writer (GL_NVENC=1) ---------------------------------------

    FFMPEG = Path(os.environ.get("GL_FFMPEG", Path(__file__).resolve().parents[1] / ".tools" / "ffmpeg-n8.1" /
                                 "bin" / "ffmpeg"))
    # Manim's PyAV writer converts RGBA to yuv420p with swscale (BT.601,
    # limited range, bilinear) and tags nothing; this does the same, so a
    # clip decodes to the same colours whichever writer made it
    # (build/nvenc_bench.py, 0 levels apart on the palette). GL_NVENC_CSC=gpu
    # hands RGBA to NVENC instead: same matrix, up to 2 levels apart, less CPU.
    _CSC = ("scale=flags=bilinear:out_color_matrix=bt601:out_range=tv,format=yuv420p,"
            "setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown")

    class NvencWriter(SceneFileWriter):
        """A SceneFileWriter that encodes with h264_nvenc in one ffmpeg process
        per scene (video/.tools/), fed raw RGBA through a pipe by Manim's queue
        and writer thread. Opening an NVENC session costs about 1.5 s here
        (WSL), so there are no partial movie files: every play streams into
        the one encoder, which writes the scene's movie file itself."""

        _ffmpeg = None

        def begin_animation(self, allow_write: bool = False, file_path=None) -> None:
            if config["write_to_movie"] and allow_write and self._ffmpeg is None:
                self._open(self.movie_file_path)

        def end_animation(self, allow_write: bool = False) -> None:
            pass                                    # one stream for the whole scene

        def _open(self, path) -> None:
            import subprocess
            from queue import Queue
            from threading import Thread

            W, H = config["pixel_width"], config["pixel_height"]
            gpu_csc = os.environ.get("GL_NVENC_CSC", "") == "gpu"
            cmd = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
                   "-s", f"{W}x{H}", "-r", f"{config['frame_rate']:g}", "-i", "-", "-an",
                   *([] if gpu_csc else ["-vf", _CSC]),
                   "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                   "-cq", os.environ.get("GL_NVENC_CQ", "23"), "-b:v", "0", str(path)]
            self._path = path
            self._ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self.queue = Queue()
            self.writer_thread = Thread(target=self.listen_and_write, args=())
            self.writer_thread.start()

        def encode_and_write_frame(self, frame, num_frames: int) -> None:
            data = np.ascontiguousarray(frame).tobytes()
            for _ in range(num_frames):
                self._ffmpeg.stdin.write(data)

        def finish(self) -> None:
            if self._ffmpeg is not None:
                self.queue.put((-1, None))
                self.writer_thread.join()
                self._ffmpeg.stdin.close()
                if self._ffmpeg.wait():
                    raise RuntimeError(f"ffmpeg (h264_nvenc) failed on {self._path}")
                self._ffmpeg = None
                self.print_file_ready_message(self._path)
            elif not config["write_to_movie"]:
                super().finish()

    def swap_renderer(renderer):
        """The CLI builds a plain OpenGLRenderer (no GL context yet) before it
        imports the scene; replace it with a GLRenderer. Built here, after
        the import, its camera also sees the frame shape the scene set
        (style.py's 9:16 frame). GL_NVENC=1 also swaps the writer."""
        if renderer is None or type(renderer) is OpenGLRenderer:
            skip = bool(getattr(renderer, "_original_skipping_status", False))
            writer = NvencWriter if os.environ.get("GL_NVENC") == "1" else SceneFileWriter
            return GLRenderer(file_writer_class=writer, skip_animations=skip)
        return renderer
