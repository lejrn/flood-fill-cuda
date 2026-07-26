# Scan-based multi-blob discovery (early attempt)

An early, unfinished attempt at the problem the chapter narrative's own
"open problems" lists still flag as unsolved: discovering multiple
blobs' seeds by scanning the image, rather than being handed seeds (as
every numbered chapter is). See `chapters/README.md`'s Chapter 4 open
problems — "Finding the seeds... connected-component labeling."

`scan_only.py` scans an image for red pixels using a CUDA kernel with a
nested/dynamic kernel launch per discovered blob (`flood_fill[1, 256](...)`
called from inside `process_image_kernel`), recoloring each blob and
counting them. It has no correctness tests, no benchmarks, and writes
output to a CWD-relative `./images/` path rather than the
`__file__`-relative convention the numbered chapters use — it predates
this project's later rigor and is kept here as a labeled starting point
for that still-open problem, not as a chapter.

Run (from the repo root, so its `./images/` output path resolves):

```bash
uv run python src/flood_fill_cuda/experiments/scan_multi_blob/scan_only.py
```
