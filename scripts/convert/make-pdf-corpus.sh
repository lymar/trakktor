#!/usr/bin/env bash
# Generates the `convert pdf` corpus: four one-page PDFs of the same page,
# differing only in how — or whether — its text is in the file. Invented here
# and generated from the LaTeX source below, so the measurements that cite them
# can be reproduced by anyone.
#
#   usage: scripts/convert/make-pdf-corpus.sh <output-dir>
#
# Needs pdflatex (TeX Live or MacTeX) and Ghostscript on PATH.
set -euo pipefail

out=${1:?usage: make-pdf-corpus.sh <output-dir>}
mkdir -p "$out"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# The page: ligatures, an en dash, curly quotes, two accented transliterations
# and a display formula — one of every class of character that a font encoding
# can lose.
cat > "$work/body.tex" <<'TEX'
\pagestyle{empty}
\begin{document}
\section*{The Difficult Office}

The staff of the office find it difficult to fulfil the different affidavits
they file, and the first of these---the one affirming that the ceiling is
sufficiently affixed---is the most baffling of all.  ``Sufficient,'' the
officer says, ``is a fine word,'' and the fifty-first clause (1841--1868)
offers no definition of it.

The transliterated name is written \=Alaska in the older sources and
Al\'aska in the newer ones; the difference is not significant.

For every real $x$ we have
\[
  \left\lfloor x \right\rfloor \leq x \leq \left\lceil x \right\rceil ,
  \qquad
  a \cdot b - c \geq 0 .
\]
\end{document}
TEX

# typeset.pdf — the default Computer Modern Type 1 fonts with the /ToUnicode
# map switched **off**, so the encoding lives inside the font program and
# nowhere in the font dictionary. That is the defect the encoding repair exists
# for, and how every PDF from an older TeX arrives; today's pdfTeX writes the
# map by default, which is why it has to be turned off on purpose here.
{ printf '\\documentclass[11pt]{article}\n\\pdfgentounicode=0\n'
  cat "$work/body.tex"; } > "$work/typeset.tex"

# plain.pdf — the same page with the map left on, so that "the text layer is
# right" has a control to be measured against.
{ printf '\\documentclass[11pt]{article}\n'
  printf '\\input{glyphtounicode}\n\\pdfgentounicode=1\n'
  cat "$work/body.tex"; } > "$work/plain.tex"

# A corpus set is fixed for as long as a measurement cites it, so the build must
# not carry today's date into the file.
export SOURCE_DATE_EPOCH=0 FORCE_SOURCE_DATE=1
for name in typeset plain; do
    (cd "$work" && pdflatex -interaction=nonstopmode -halt-on-error \
        "$name.tex" >/dev/null)
    cp "$work/$name.pdf" "$out/$name.pdf"
done

# outlined.pdf — every glyph converted to vector outlines: letters on the page,
# not one character in the file.
gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dNoOutputFonts \
   -dOmitInfoDate=true -dOmitID=true -dOmitXMP=true \
   -sOutputFile="$out/outlined.pdf" "$work/plain.pdf"

# scanned.pdf — the page rasterized and wrapped back up, which is what comes
# out of a scanner.
gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r150 \
   -sOutputFile="$work/scan.png" "$work/plain.pdf"
python3 - "$work/scan.png" "$out/scanned.pdf" <<'PY'
"""Wraps a PNG as the single page of a PDF, reusing its own zlib stream."""
import struct
import sys

png, out = sys.argv[1], sys.argv[2]
data = open(png, 'rb').read()
width, height = struct.unpack('>II', data[16:24])

chunks, at = [], 8
while at < len(data):
    length = struct.unpack('>I', data[at:at + 4])[0]
    kind = data[at + 4:at + 8]
    if kind == b'IDAT':
        chunks.append(data[at + 8:at + 8 + length])
    at += 12 + length
idat = b''.join(chunks)

content = b'q 612 0 0 792 0 0 cm /Im0 Do Q\n'
objects = [
    b'<< /Type /Catalog /Pages 2 0 R >>',
    b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
    b'/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>',
    b'<< /Length %d >>\nstream\n%sendstream' % (len(content), content),
    b'<< /Type /XObject /Subtype /Image /Width %d /Height %d '
    b'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode '
    b'/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 '
    b'/Columns %d >> /Length %d >>\nstream\n%s\nendstream'
    % (width, height, width, len(idat), idat),
]

pdf, offsets = b'%PDF-1.4\n', []
for number, body in enumerate(objects, 1):
    offsets.append(len(pdf))
    pdf += b'%d 0 obj\n%s\nendobj\n' % (number, body)
xref = len(pdf)
pdf += b'xref\n0 %d\n0000000000 65535 f \n' % (len(objects) + 1)
for offset in offsets:
    pdf += b'%010d 00000 n \n' % offset
pdf += b'trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n' % (
    len(objects) + 1, xref)
open(out, 'wb').write(pdf)
PY

ls -l "$out"
