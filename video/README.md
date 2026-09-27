# flood-fill-cuda explainer video

Three panes for the whole video: the benchmark matrix (17 shapes, one
column per chapter, a shape glows when the GPU beats the CPU), the
chapter's blob (finished chapters sweep up into a strip), and the GPU
schematic (SMs, blocks, threads, memory per chapter). Manim scenes, local
Kokoro narration, ffmpeg assembly. About 96 s, 1920x1080.

- `scenes/BRIEF.md`: what is on screen and the rules.
- `HANDOFF.md`: stack, pipeline, commands, gotchas, open items.
- Every number comes from a committed benchmark JSON via
  `scenes/panes/data.py`; `build/audit_numbers.py` checks them.
