# flood-fill-cuda explainer video

A 30 s intro states the problem (a defence camera labelling drones in
every frame, real time vs 24 s per frame, on drone-show footage), then
three panes for the rest of the video: the benchmark matrix (17 shapes,
one column per chapter, a shape glows when the GPU beats the CPU), the
chapter's blob (finished chapters sweep up into a strip), and the GPU
schematic (SMs, blocks, threads, memory per chapter). Manim scenes,
ffmpeg assembly.

Two layouts and two voices make four cuts, in `out/` (gitignored):

| file | frame | voice | length |
|---|---|---|---|
| `final_landscape_kokoro.mp4` | 1920x1080 | Kokoro `am_adam`, local | 134.5 s |
| `final_landscape_elevenlabs.mp4` | 1920x1080 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s |
| `final_vertical_kokoro.mp4` | 1080x1920 | Kokoro `am_adam`, local | 134.5 s |
| `final_vertical_elevenlabs.mp4` | 1080x1920 | ElevenLabs "Peter Baker", `eleven_v4` | 185.7 s |

- `scenes/BRIEF.md`: what is on screen and the rules.
- `HANDOFF.md`: stack, pipeline, commands, gotchas, open items.
- Every number comes from a committed benchmark JSON via
  `scenes/panes/data.py`; `build/audit_numbers.py` checks them.
