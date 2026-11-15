#!/usr/bin/env bash
# Builds the distributables into dist/:
#   hwqa-<ver>-py3-none-any.whl   pip / pipx install
#   hwqa.pyz                      single file, runs with any python3 >= 3.9 (no install)
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist build ./*.egg-info && mkdir -p dist

python3 -m pip wheel --no-deps --no-build-isolation -w dist . >/dev/null 2>&1 \
  || python3 -m pip wheel --no-deps -w dist .

stage="$(mktemp -d)"
cp -r hwqa "$stage/"
find "$stage" -name __pycache__ -prune -exec rm -rf {} +
python3 -m zipapp "$stage" -m "hwqa.cli:main" -p "/usr/bin/env python3" -o dist/hwqa.pyz
chmod +x dist/hwqa.pyz
rm -rf "$stage" build ./*.egg-info

(cd dist && sha256sum ./* > SHA256SUMS)
ls -l dist
