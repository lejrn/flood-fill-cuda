# flood-fill-video

A 30-60 second explainer for the parent project, rendered entirely
with free, project-local tools.

- `scenes/` Manim scenes, one file per beat
- `narration/` script text and generated voice tracks
- `assets/` frame sequences extracted from the parent repo's wavefront GIFs
- `build/` assembly scripts (ffmpeg concat, 16:9 and 9:16 layouts)

Setup: `uv sync`. System headers needed once for ManimPango:
`libcairo2-dev libpango1.0-dev`.
