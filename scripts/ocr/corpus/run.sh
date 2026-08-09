#!/usr/bin/env bash
# Read every frame of the photographed-page corpus in three configurations and
# put the readings where `measure.py` can score them.
#
#   run.sh <corpus dir> <output dir> [trakktor args...]
#
# The three are the question the corpus exists to answer:
#
#   plain    — no preprocessing at all, the floor everything is compared to
#   upstream — what PaddleOCR's own pre-processor does: orientation, unwarp
#   full     — the same with trakktor's sheet finder in front of the unwarper
#
# The flat 300 dpi render of the same PDF is read too, under the name `flat`:
# it is the scan the photographs are measured against.

set -euo pipefail

corpus=${1:?corpus directory}
out=${2:?output directory}
shift 2

trakktor=${TRAKKTOR:-target/release/trakktor}
mkdir -p "$out"/{plain,upstream,full,rectified}

for page in "$corpus"/*.jpg; do
    name=$(basename "$page" .jpg)
    echo "== $name"
    "$trakktor" ocr paddle "$page" --text "$@" > "$out/plain/$name.txt"
    "$trakktor" ocr paddle "$page" --text --doc-orientation --unwarp "$@" \
        > "$out/upstream/$name.txt"
    "$trakktor" ocr paddle "$page" --text --doc-orientation --sheet --unwarp \
        --rectified "$out/rectified/$name.png" "$@" > "$out/full/$name.txt"
done

echo "== flat"
"$trakktor" ocr paddle "$corpus/flat.png" --text "$@" > "$out/plain/flat.txt"
cp "$out/plain/flat.txt" "$out/upstream/flat.txt"
cp "$out/plain/flat.txt" "$out/full/flat.txt"
