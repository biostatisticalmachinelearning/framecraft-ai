# Data Layout

Local data lives here and is ignored by git.

Recommended structure:
- `data/raw/` Downloaded source movies
- `data/frames/` Extracted frames per movie
- `data/manifests/` JSONL manifests for training/validation/testing
- `data/registry.jsonl` Source registry with hashes and license metadata

Notes:
- Keep `data/registry.jsonl` under 100MB; archive older entries if needed.
- Store large datasets on the desktop machine; copy only small subsets locally.
