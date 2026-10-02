"""One full cut per Kokoro voice, so the voice can be chosen by ear.

    uv run build/variants.py                 # every male voice, deepest first
    uv run build/variants.py am_onyx bm_lewis

Per voice: narration into out/kokoro_<voice>/, a full render (clip
lengths follow the narration), assembly with the seam check, and
out/final_landscape_kokoro_<voice>.mp4. About 6 minutes per voice on this
laptop; run it alone (the TTS needs ~2 GB of RAM).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

VIDEO = Path(__file__).resolve().parents[1]
PY = VIDEO / ".venv" / "bin" / "python"
# the venv first on PATH (misaki shells out to `python`), CPU-only torch (see HANDOFF)
ENV = dict(os.environ, PATH=f"{VIDEO / '.venv' / 'bin'}:{os.environ.get('PATH', '')}", CUDA_VISIBLE_DEVICES="")

# Kokoro v1.0 male voices, deepest first (median pitch, narration/pick_voice.py)
MALE_VOICES = ["am_onyx", "bm_lewis", "am_echo", "am_adam", "am_michael", "bm_daniel",
               "bm_george", "am_fenrir", "am_eric", "am_liam", "am_puck", "bm_fable", "am_santa"]


def main() -> int:
    voices = sys.argv[1:] or MALE_VOICES
    for voice in voices:
        t0 = time.time()
        name = f"kokoro_{voice}"
        print(f"== {voice}", flush=True)
        subprocess.run([str(PY), str(VIDEO / "narration" / "tts_kokoro.py"), "--voice", voice, "--out", name],
                       cwd=VIDEO, check=True, env=ENV)
        subprocess.run([str(PY), str(VIDEO / "build" / "assemble.py"), "--render", "--voice", name, "--check"],
                       cwd=VIDEO, check=True)
        print(f"== {voice} done in {(time.time() - t0) / 60:.1f} min -> out/final_landscape_{name}.mp4",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
