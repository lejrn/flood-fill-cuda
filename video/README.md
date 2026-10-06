# flood-fill-cuda explainer video

A 30 s intro states the problem (a defence camera labelling drones in
every frame, real time vs 24 s per frame, on drone-show footage), then
three panes for the rest of the video: the benchmark matrix (17 shapes,
one column per chapter, a shape glows when the GPU beats the CPU), the
chapter's blob (finished chapters sweep up into a strip), and the GPU
schematic (SMs, blocks, threads, memory per chapter). Before the outro,
one beat on the Triton twins: every chapter rebuilt in Triton, and the
matrix flips to Numba time ÷ Triton time from one session. Manim scenes,
ffmpeg assembly.

Two layouts and two voices make four cuts, in `out/` (gitignored):

| file | frame | voice | length | Triton beat |
|---|---|---|---|---|
| `final_landscape_kokoro.mp4` | 1920x1080 | Kokoro `am_adam`, local | 134.5 s | not yet |
| `final_landscape_elevenlabs.mp4` | 1920x1080 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s | not yet |
| `final_vertical_kokoro.mp4` | 1080x1920 | Kokoro `am_adam`, local | 146.5 s | yes |
| `final_vertical_elevenlabs.mp4` | 1080x1920 | ElevenLabs "Peter Baker", `eleven_v4` | 200.9 s | yes |

The landscape cuts were rendered before the Triton beat existed. The
scenes render it in both layouts, so a landscape re-render picks it up.

- `scenes/BRIEF.md`: what is on screen and the rules.
- `HANDOFF.md`: stack, pipeline, commands, gotchas, open items.
- Every number comes from a committed benchmark JSON via
  `scenes/panes/data.py`; `build/audit_numbers.py` checks them.
