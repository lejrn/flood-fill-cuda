# Benchmark inputs

The two images every chapter's headline numbers are measured on. They are
synthetic: pure red `(255, 0, 0)` shapes on white `(255, 255, 255)`, two
colours only, no anti-aliasing.

| file | size | red pixels | blobs (8-connectivity) |
|---|---|---|---|
| `input_blobs.png` | 9000 x 9000 (81 Mpx) | 13,451,960 | 2,522 |
| `input_blocks.png` | 1000 x 1000 (1 Mpx) | 387,587 | 21,618 |

The blob counts are the CPU oracle's and OpenCV's
(`src/flood_fill_cuda/shared/test_oracles_vs_opencv.py` checks both
files).

Checksums (the benchmark JSONs of ch08 record the same MD5s as
`png_md5`):

```
SHA-256  6abb863d7fe7a2b2c809f3b7edadced14d08580fbabacf5bd466eeb61624bc3d  input_blobs.png
SHA-256  462a0a748afb336777c66ac060e9f7c199148d63932bec3be30bafa0642f2e4d  input_blocks.png
MD5      47639eb37f973b5092ec2c4db86f45a4  input_blobs.png
MD5      68f1fda7f2eba3d73b87e1cc11155061  input_blocks.png
```

These files are the reference. `utils/generate_blobs_image.py` and
`utils/generate_blocks_image.py` draw images of this kind, but they do
not seed their random generator, so they cannot reproduce these exact
files.
