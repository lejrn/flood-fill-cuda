# Project page

A one-page site for flood-fill-cuda in the style of the
[Nerfies](https://nerfies.github.io) project page. It covers the Numba
chapters (ch01-ch06). The Triton twins are not on it yet.

## View it

Open `index.html` in a browser. It needs no build step and no server.

Links to the repo (the story, the code, the results) are written as
relative paths, so they work in a local clone. When the page is served
from GitHub Pages (`<owner>.github.io/<repo>/...`), `static/js/index.js`
points them at github.com instead. The account name is read from the
address bar, so it is never written into this folder.

## Layout

```
index.html                 the page
static/css/                Bulma 0.9.1, bulma-carousel, bulma-slider (MIT), index.css
static/js/                 bulma-carousel, bulma-slider, index.js (no jQuery)
static/images/             favicon, posters, the ch06 before/after still
static/videos/             teaser, carousel clips, the two side-by-side clips, explainer
static/scrub/              frame sequences for the two scrub sliders
tools/                     scripts that rebuild everything above from the repo
```

## Rebuild

Every asset comes from a committed result or from the video pipeline.

| what | script | source |
|---|---|---|
| inline figures in `index.html` | `tools/build_figures.py` | `results/ch06_gpu_nblob_runs/figures/*.svg` |
| teaser and carousel clips | `tools/make_clips.sh` | the chapters' wavefront GIFs, the video's `s07_runs` clip |
| scrub frames | `tools/make_scrub_frames.py` | ch01 and ch05 wavefront GIFs |
| explainer web cut | `tools/make_explainer.sh` | `video/out/final_landscape_elevenlabs.mp4` |

Run them from the repo root:

```
python project-page/tools/build_figures.py
FFMPEG=video/.venv/bin/ffmpeg PYTHON=.venv/bin/python bash project-page/tools/make_clips.sh
.venv/bin/python project-page/tools/make_scrub_frames.py
FFMPEG=video/.venv/bin/ffmpeg bash project-page/tools/make_explainer.sh
```

`video/out` and `video/media` are gitignored, so the clip and explainer
scripts need a checkout where the video has been rendered. Point
`VIDEO_DIR` at it if it lives elsewhere.

The figures are inlined rather than linked so that the page's light and
dark colours reach them. The script swaps the seven hard-coded colours for
CSS classes and changes nothing else.

## Notes on the content

- The explainer starts at the first chapter. The first 44 s of the full
  cut show third-party drone-show footage with no recorded licence, so
  the page leaves them out.
- Chapters 1-4 on the 2,522-blob image are estimates (a six-blob sample
  times 2,522 launches), and the page says so.
- The ch06 numbers and the pure-Python and @njit numbers come from
  different sessions. The page pairs same-session numbers where it
  compares them.

## Licence

The design and code are adapted from the Nerfies page, which is CC BY-SA
4.0. This page is shared under the same licence, with the link back in
its footer.
