# third_party/

Upstream code this repo is built from. `repos.yaml` pins every repo to a commit (with license and
what we use it for); `python third_party/fetch.py` re-creates the checkouts. The checkouts are
git-ignored, so only `repos.yaml`, `fetch.py` and this file are tracked.

Convention: TTA methods and benchmark code are implemented from these checkouts rather than from
memory. A module ported into `src/` keeps the upstream license header and names the upstream file
and commit in its docstring.

Facts read from these repos that the plan relies on:

- **CCC** is generated locally from ImageNet *val* (`CCC/generate.py`); the hosted endpoint is gone.
  Each stream is 7.5M JPEG-85 images in webdataset shards (`serial_*.tar`, 25,000 images each).
  Difficulty = `--baseline` 0/20/40 (Hard/Medium/Easy); transition speeds 1000/2000/5000; seeds 43/44/45.
  Generation needs `webdataset`, `wand` (ImageMagick), `opencv`, `scikit-image`, `numba`, and downloads
  `ccc_accuracy_matrix.pickle` from a University of Tuebingen URL (so it needs internet once).
- **RDumb** (`CCC/models/rdumb.py`) is ETA with a reset to the source weights *and optimizer state*
  every 1000 steps (batch 64), SGD lr 2.5e-4, momentum 0.9, e_margin = 0.4 ln(1000), d_margin 0.05.
- **ImageNet-C and CCC images are already 224x224**; the benchmark repo evaluates them with no resize.
- **CoTTA's 10 orders** are `cotta/{cifar,imagenet}/cfgs/10orders/`.
- **ImageNet-R/A** use 200-class subsets scored with `logits[:, mask]`.
- **ImageNet-V2** folders are named by class index (0..999); torchvision's `ImageFolder` would sort
  them lexicographically ('0', '1', '10', '100', ...) and mislabel everything.
