#!/usr/bin/env bash
set -euo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git clone https://github.com/IIIS-Li-Group/OpenFE.git "$H/baselines/vendor/OpenFE"
git -C "$H/baselines/vendor/OpenFE" checkout 3e3bdf9
