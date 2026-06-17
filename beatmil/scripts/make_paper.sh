#!/usr/bin/env bash
# Compile paper.tex -> paper.pdf with bibliography.
#
# Usage:
#   bash make_paper.sh                # compile in ~/beatmil/paper/
#   PAPER=/custom/path bash make_paper.sh

set -e
PAPER="${PAPER:-$HOME/beatmil/paper}"
SRC="${SRC:-$HOME/beatmil/src}"

# Ensure paper dir exists with required files
mkdir -p "$PAPER/figures" "$PAPER/tables"

# Copy paper.tex and refs.bib from src if not already in paper dir
if [ ! -f "$PAPER/paper.tex" ]; then
    cp "$HOME/beatmil/paper.tex" "$PAPER/paper.tex"
fi
if [ ! -f "$PAPER/refs.bib" ]; then
    cp "$HOME/beatmil/refs.bib" "$PAPER/refs.bib"
fi

# Link or copy figures
if [ -d "$HOME/beatmil/figures" ]; then
    for f in "$HOME/beatmil/figures"/*.png "$HOME/beatmil/figures"/*.pdf; do
        [ -f "$f" ] && cp -f "$f" "$PAPER/figures/" 2>/dev/null || true
    done
fi

# Generate the LaTeX tables from eval JSONs
cd "$SRC"
python tables.py

# Compile (LaTeX -> BibTeX -> LaTeX -> LaTeX is the standard 4-pass sequence)
cd "$PAPER"
echo
echo "[make_paper] pass 1 (pdflatex)"
pdflatex -interaction=nonstopmode paper.tex > /tmp/latex_pass1.log 2>&1 || true

echo "[make_paper] pass 2 (bibtex)"
bibtex paper > /tmp/bibtex.log 2>&1 || {
    echo "  bibtex failed — check /tmp/bibtex.log"
    cat /tmp/bibtex.log
}

echo "[make_paper] pass 3 (pdflatex)"
pdflatex -interaction=nonstopmode paper.tex > /tmp/latex_pass3.log 2>&1 || true

echo "[make_paper] pass 4 (pdflatex)"
pdflatex -interaction=nonstopmode paper.tex > /tmp/latex_pass4.log 2>&1 || true

# Report
if [ -f "paper.pdf" ]; then
    PAGES=$(pdfinfo paper.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    SIZE=$(du -h paper.pdf | awk '{print $1}')
    echo
    echo "[make_paper] ✓ paper.pdf built: $PAGES pages, $SIZE"
    echo "[make_paper] location: $PAPER/paper.pdf"
    if [ "$PAGES" -gt 6 ]; then
        echo "[make_paper] ⚠ PAPER EXCEEDS 6-PAGE BMEiCON LIMIT — trim Related Work or Discussion"
    fi
    # Warnings summary
    if grep -q "Warning" /tmp/latex_pass4.log; then
        echo
        echo "[make_paper] LaTeX warnings:"
        grep -E "Warning|Undefined" /tmp/latex_pass4.log | head -20
    fi
else
    echo "[make_paper] ✗ paper.pdf NOT built — check /tmp/latex_pass*.log"
    tail -30 /tmp/latex_pass1.log
    exit 1
fi
