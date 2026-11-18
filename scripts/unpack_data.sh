#!/usr/bin/env bash
# Check the data archives against their checksums and unpack them to results/.
#
#   scripts/unpack_data.sh                 all archives
#   scripts/unpack_data.sh counting-data   only the named ones
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v sha256sum >/dev/null 2>&1; then
  (cd data && sha256sum -c SHA256SUMS)
else
  (cd data && shasum -a 256 -c SHA256SUMS)
fi

if [ "$#" -gt 0 ]; then
  for name in "$@"; do
    tar -xJf "data/$name.tar.xz"
  done
else
  for archive in data/*.tar.xz; do
    tar -xJf "$archive"
  done
fi
echo "unpacked to results/"
