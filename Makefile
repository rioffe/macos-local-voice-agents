# Makefile - build the architecture documentation PDF.
#
# Diagrams are rendered via the vendored ./md2pdf.sh (a pandoc + xelatex +
# mermaid-filter wrapper). The script now defaults --mermaid to *vector* output
# (crisp at any zoom), so you only add --mermaid/--toc/--margin for layout.
#
# Usage:
#   make            # build docs/ARCHITECTURE.pdf
#   make architecture
#   make MARGIN=0.3in     # override the page margin
#   make clean

# --- paths / flags ---------------------------------------------------------
MD2PDF      := sh md2pdf.sh
DOC         := docs/ARCHITECTURE.md
PDF         := docs/ARCHITECTURE.pdf
MARGIN      ?= 0.5in

# Mermaid quality is now baked into md2pdf.sh (vector by default); keep these
# explicit here only so the build command is self-documenting. Export them so
# they reach the mermaid-filter subprocess even when the script's own defaults
# would otherwise apply.
MERMAID_FILTER_FORMAT ?= pdf
MERMAID_FILTER_SCALE  ?= 3
export MERMAID_FILTER_FORMAT MERMAID_FILTER_SCALE

.PHONY: all build architecture clean

all: build

build architecture: $(PDF)

$(PDF): $(DOC) md2pdf.sh
	$(MD2PDF) --toc --mermaid --margin $(MARGIN) $(DOC)

clean:
	rm -f $(PDF)
