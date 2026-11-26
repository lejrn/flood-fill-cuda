# Project page

A one-page site for flood-fill-cuda in the style of the
[Nerfies](https://nerfies.github.io) project page. It tells the BFS
story, from one pixel per step (ch00-ch05) to tiles (ch07, ch08), then
the CPU on every core (ch09) and the session against OpenCV and CuPy.
ch06, which is not a BFS, has a short reference section. The Triton
twins are not on it yet.

## View it

Open `index.html` in a browser. It needs no build step and no server.

Links to the repo (the story, the code, the results) are written as
relative paths, so they work in a local clone. When the page is served
from GitHub Pages at `<owner>.github.io/<repo>/`, `static/js/index.js`
points them at github.com instead. The account name is read from the
address bar, so it is never written into this folder.

## Publish

`.github/workflows/pages.yml` publishes this folder to GitHub Pages at
`<owner>.github.io/<repo>/`. It runs on every push to `main` that touches
`project-page/`, and by hand from the Actions tab ("Project page", Run
workflow). The repo's Settings, Pages, Source is set to "GitHub Actions".

GitHub Pages can publish a branch only from the repo root or `/docs`,
which is why a workflow uploads this folder instead.

On a custom domain or a user site the relative `../` links stay as they
are and lead nowhere, so point them at github.com by hand there.

The workflow publishes from a clean checkout. Do not publish by copying
this folder from disk: a local `tools/__pycache__/` holds bytecode with
local paths in it.

## Layout

```
index.html                 the page
THIRD_PARTY_NOTICES.md     licenses of the bundled CSS, JS and icons
static/css/                Bulma 0.9.1, bulma-carousel, bulma-slider (MIT), index.css
static/js/                 bulma-carousel, index.js (no jQuery)
static/images/             favicon, posters, the ch06 before/after still
static/videos/             teaser, carousel clips, side-by-side clips, the race, the tile clips,
                           the narrated explainer and its captions
static/scrub/              frame sequence for the comb scrub slider
tools/                     scripts that rebuild everything above from the repo
```

## Rebuild

Every asset comes from a committed result or from the video pipeline.

| what | script | source |
|---|---|---|
| the chain chart's SVG (`tools/chain_bfs.svg`) | `tools/make_chain_bfs.py` | the newest overview JSON, ch08 session `dense_20261016T054922Z` |
| inline figures in `index.html` | `tools/build_figures.py` | `tools/chain_bfs.svg`, `results/ch06_gpu_nblob_runs/figures/runs_vs_pixels.svg` |
| teaser, carousel and side-by-side clips | `tools/make_clips.sh` | the chapters' wavefront GIFs |
| comb scrub frames | `tools/make_scrub_frames.py` | ch05 comb wavefront GIFs |
| CPU vs GPU race | `tools/make_race.py` (data in `tools/race_spec.json`) | ch03 benchmark JSON, scene `sq_8000_center` |
| tile clips of ch07 and ch08, sections and carousel | `tools/make_tile_clips.py` | `video/assets/tiles/*.npz`, recorded from the kernels |
| explainer web cut and poster | `tools/make_explainer.sh` | `video/out/final_chain_iso_gl_landscape_kokoro_paced.mp4` |
| captions and transcript of the explainer | `tools/make_captions.py` | `video/narration/script_chain.md` and the take's word times in `video/out/` |

Run them from the repo root:

```
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python project-page/tools/make_chain_bfs.py
python project-page/tools/build_figures.py
FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python bash project-page/tools/make_clips.sh
.venv/bin/python project-page/tools/make_scrub_frames.py
FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_race.py
FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_tile_clips.py
FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python bash project-page/tools/make_explainer.sh
FFMPEG=video/.venv/bin/ffmpeg .venv/bin/python project-page/tools/make_captions.py
```

`video/out` is gitignored. Only the explainer and its captions need a
checkout where the video has been rendered; point `VIDEO_DIR` at it if
it lives elsewhere. Every other script builds from committed files.

`static/videos/explainer.en.vtt` holds the captions. `make_captions.py`
times them on the narration's own word times and checks nothing by ear;
both it and `make_explainer.sh` add the same 0.5 s lead-in.

The figures are inlined rather than linked so that the page's light and
dark colors reach them. `build_figures.py` swaps the seven hard-coded
colors for CSS classes and fits each figure to the page: it drops,
relabels or moves a few text lines (listed in its `EDITS` table), trims
the viewBox, adds ids and a text description, and replaces dashes with
hyphens. No bar, dot, line or number changes.

## Notes on the content

- The explainer is the chain cut, v5: "flood fill, from pixels to
  tiles". It has no chapter 6 in it. The older three-pane cut of
  ch01-ch06 is no longer on the page.
- The explainer opens and closes on drone-show footage by a third
  party. It is not covered by this folder's license.
- The page's tile clips are drawn from recorded data. One pane is not a
  recording: the pixel BFS beside the tile BFS is drawn from its level
  rule on a solid square, and its caption says so.
- Chapters 1-4 on the 2,522-blob image are estimates (six sample launches
  times the launch count), and the page says so.
- The video's numbers come from its own session
  (`v5_20261017T192640Z`), the page's from the chapters' sessions. Where
  they differ (2.21 ms against 2.40 ms for ch08), the page says why.
- The numbers of ch07-ch09 and of the libraries are copied by hand from
  the READMEs, which quote the committed sessions
  (`tiles_20261016T064724Z`, `dense_20261016T054922Z`,
  `ch09_20261017T233237Z`, `libs_20261125T140827Z`). Update the page
  when those change.

## License

The design and code are adapted from the Nerfies page, which is CC BY-SA
4.0. This page is shared under the same license, with the link back in
its footer. See `THIRD_PARTY_NOTICES.md` for the bundled files.

The rest of the repository is under the MIT license (`../LICENSE`).
