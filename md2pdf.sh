#!/bin/bash

# md2pdf.sh: Converts Markdown with LaTeX to PDF using pandoc and xelatex.

set -e

TOC_FLAG=()
MERMAID_FLAG=()
GEOMETRY_FLAG=()
MARGIN_HEADER_FILE=""

while [[ $# -gt 0 ]]; do
  echo "Arg: $1"
  case $1 in
     --toc)
      TOC_FLAG=(--toc)
      shift
      ;;
     --mermaid)
      MERMAID_FLAG=(--filter mermaid-filter)
      shift
      ;;
     --margin)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Error: --margin requires a value (e.g. 0.5in, 1cm, 0.3in)." >&2
        exit 1
      fi
      MARGIN_VAL="$1"
      shift
      MARGIN_HEADER_FILE=$(mktemp /tmp/md2pdf_geometry_$$_XXXXXX)
      echo "\usepackage[margin=${MARGIN_VAL}]{geometry}" > "$MARGIN_HEADER_FILE"
      GEOMETRY_FLAG=(--include-in-header="$MARGIN_HEADER_FILE")
      ;;
     *)
      INPUT_FILE="$1"
      shift
      ;;
  esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Usage: $0 [--toc] [--mermaid] [--margin MARGIN] <input_file.md>"
    echo "    --margin MARGIN   Set page margins on all sides (e.g. 0.5in, 1cm, 0.3in)."
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# When rendering mermaid, 'mermaid-filter' -> 'mmdc' -> puppeteer needs a
# Chromium executable. puppeteer's pinned rev is often missing from
# ~/.cache/puppeteer; fall back to a system Chrome/Chromium/Edge via
# PUPPETEER_EXECUTABLE_PATH (an already-exported value is always respected).
if [ "${#MERMAID_FLAG[@]}" -gt 0 ]; then
    # mermaid-filter defaults to a low-DPI PNG (800px, scale=1) that looks
    # fuzzy in PDFs. Default to vector output (crisp at any zoom) with a
    # high-scale raster fallback, but always respect a value the user set.
    if [ -z "${MERMAID_FILTER_FORMAT:-}" ]; then
        MERMAID_FILTER_FORMAT="pdf"
        export MERMAID_FILTER_FORMAT
        echo "mermaid: defaulting MERMAID_FILTER_FORMAT=pdf (vector; crisp at any zoom)"
    fi
    if [ -z "${MERMAID_FILTER_SCALE:-}" ]; then
        MERMAID_FILTER_SCALE="3"
        export MERMAID_FILTER_SCALE
        echo "mermaid: defaulting MERMAID_FILTER_SCALE=3 (high-res raster fallback)"
    fi
    # mmdc -> puppeteer needs a Chromium binary. A pinned rev is often
    # missing from ~/.cache/puppeteer, so fall back to a system browser.
    # An already-exported PUPPETEER_EXECUTABLE_PATH is always respected.
    if [ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
        found=0
        if [ "$(uname)" = "Darwin" ]; then
            for cand in \
                            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
                            "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary" \
                            "/Applications/Chromium.app/Contents/MacOS/Chromium" \
                            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"; do
                if [ -x "$cand" ]; then
                    PUPPETEER_EXECUTABLE_PATH="$cand"
                    export PUPPETEER_EXECUTABLE_PATH
                    found=1
                    break
                fi
            done
        else
            for cand in google-chrome google-chrome-stable chromium chromium-browser; do
                if bin="$(command -v "$cand" 2>/dev/null)"; then
                    PUPPETEER_EXECUTABLE_PATH="$bin"
                    export PUPPETEER_EXECUTABLE_PATH
                    found=1
                    break
                fi
            done
        fi
        if [ "$found" -eq 1 ]; then
            echo "mermaid: using system browser via PUPPETEER_EXECUTABLE_PATH=$PUPPETEER_EXECUTABLE_PATH"
        else
            echo "Warning: no Chromium/Chrome found for mermaid; diagrams may fail to render." >&2
            echo "           (set PUPPETEER_EXECUTABLE_PATH, or run \"npx puppeteer browsers install chrome\")." >&2
        fi
    else
        echo "mermaid: using PUPPETEER_EXECUTABLE_PATH=$PUPPETEER_EXECUTABLE_PATH"
    fi
fi

OUTPUT_FILE="${INPUT_FILE%.md}.pdf"
TEMP_FILE=$(mktemp /tmp/md2pdf.XXXXXX.md)

echo "Processing '$INPUT_FILE'..."

# Pre-process to convert [ ... ] math blocks to $$ ... $$
# We use a more robust regex to handle potential whitespace
sed -e 's/^[[:space:]]*\[[[:space:]]*$/$$\n/' \
     -e 's/^[[:space:]]*\][[:space:]]*$/\n$$/' \
     "$INPUT_FILE" > "$TEMP_FILE"

echo "Converting to PDF via pandoc (using xelatex) with ${TOC_FLAG[*]} ${MERMAID_FLAG[*]} ${GEOMETRY_FLAG[*]}..."

if pandoc "$TEMP_FILE" "${TOC_FLAG[@]}" "${MERMAID_FLAG[@]}" "${GEOMETRY_FLAG[@]}" --pdf-engine=xelatex -o "$OUTPUT_FILE"; then
    echo "Success! Created '$OUTPUT_FILE'."
else
    # Fallback to default engine if xelatex fails
    echo "xelatex failed or not found. Retrying with default engine..."
    if pandoc "$TEMP_FILE" "${TOC_FLAG[@]}" "${MERMAID_FLAG[@]}" "${GEOMETRY_FLAG[@]}" -o "$OUTPUT_FILE"; then
        echo "Success! Created '$OUTPUT_FILE' (using default engine)."
    else
        echo "Error: Conversion failed."
        rm -f "$TEMP_FILE" "$MARGIN_HEADER_FILE"
        exit 1
    fi
fi

rm -f "$TEMP_FILE" "$MARGIN_HEADER_FILE"
