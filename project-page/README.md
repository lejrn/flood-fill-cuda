# Project page

A one-page site for flood-fill-cuda in the style of the
[Nerfies](https://nerfies.github.io) project page. It covers the Numba
chapters (ch01-ch06). The Triton twins are not on it yet.

## View it

Open `index.html` in a browser. It needs no build step and no server.

Links to the repo (the story, the code, the results) are written as
relative paths, so they work in a local clone. When the page is served
from GitHub Pages at `<owner>.github.io/<repo>/`, `static/js/index.js`
points them at github.com instead. The account name is read from the
address bar, so it is never written into this folder.

## Publish

GitHub Pages can publish a branch only from the repo root or `/docs`, and
this page lives in `project-page/`. Publish it with a GitHub Actions
workflow instead:

1. Add `.github/workflows/pages.yml` that runs on pushes to the default
   branch touching `project-page/**`, with `contents: read`,
   `pages: write` and `id-token: write`. Its steps are
   `actions/checkout`, `actions/configure-pages`,
   `actions/upload-pages-artifact` with `path: project-page`, and
   `actions/deploy-pages`.
2. In Settings, Pages, set Source to "GitHub Actions".

The site then lives at `<owner>.github.io/<repo>/`, the shape the link
rewriting expects. On a custom domain or a user site the relative `../`
links stay as they are and lead nowhere, so point them at github.com by
hand there.

Publish from a clean checkout, not by copying this folder from disk: a
local `tools/__pycache__/` holds bytecode with local paths in it.

## Layout

```
index.html                 the page
THIRD_PARTY_NOTICES.md     licenses of the bundled CSS, JS and icons
static/css/                Bulma 0.9.1, bulma-carousel, bulma-slider (MIT), index.css
static/js/                 bulma-carousel, index.js (no jQuery)
static/images/             favicon, posters, the ch06 before/after still
static/videos/             teaser, carousel clips, the two side-by-side clips, explainer and its captions
static/scrub/              frame sequences for the two scrub sliders
tools/                     scripts that rebuild everything above from the repo
```

## Rebuild

Every asset comes from a committed result or from the video pipeline.

| what | script | source |
|---|---|---|
| inline figures in `index.html` | `tools/build_figures.py` | `results/ch06_gpu_nblob_runs/figures/*.svg` |
| teaser, carousel and side-by-side clips | `tools/make_clips.sh` | the chapters' wavefront GIFs, the video's `s07_runs` clip |
| scrub frames | `tools/make_scrub_frames.py` | ch01 and ch05 wavefront GIFs |
| explainer web cut | `tools/make_explainer.sh` | `video/out/final_landscape_elevenlabs.mp4` |

Run them from the repo root:

```
export PYTHONDONTWRITEBYTECODE=1
python project-page/tools/build_figures.py
FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python bash project-page/tools/make_clips.sh
.venv/bin/python project-page/tools/make_scrub_frames.py
FFMPEG=video/.venv/bin/ffmpeg bash project-page/tools/make_explainer.sh
```

`video/out` and `video/media` are gitignored. Only the ch06 "runs" clip
and the explainer need a checkout where the video has been rendered;
point `VIDEO_DIR` at it if it lives elsewhere. Without it, `make_clips.sh`
skips the runs clip and builds everything else.

`static/videos/explainer.en.vtt` holds the captions, timed against the
audio of the web cut. Rebuild them by hand if the narration changes.

The figures are inlined rather than linked so that the page's light and
dark colors reach them. `build_figures.py` swaps the seven hard-coded
colors for CSS classes and fits each figure to the page: it drops,
relabels or moves a few text lines (listed in its `EDITS` table), trims
the viewBox, adds ids and a text description, and replaces dashes with
hyphens. No bar, dot, line or number changes.

## Notes on the content

- The explainer starts at the first chapter. The first 44 s of the full
  cut show third-party drone-show footage with no recorded license, so
  the page leaves them out.
- Chapters 1-4 on the 2,522-blob image are estimates (six sample launches
  times the launch count), and the page says so.
- The ch06 numbers and the pure-Python and @njit numbers come from
  different sessions. The page pairs same-session numbers where it
  compares them.

## License

The design and code are adapted from the Nerfies page, which is CC BY-SA
4.0. This page is shared under the same license, with the link back in
its footer. See `THIRD_PARTY_NOTICES.md` for the bundled files.
