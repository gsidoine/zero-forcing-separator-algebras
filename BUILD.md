# Manuscript build

Reference build environment:

- pdfTeX 1.40.26
- TeX Live 2025/dev

Build from the package root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error zero_forcing_separator_algebras.tex
pdflatex -interaction=nonstopmode -halt-on-error zero_forcing_separator_algebras.tex
pdflatex -interaction=nonstopmode -halt-on-error zero_forcing_separator_algebras.tex
```

No BibTeX or Biber pass is required because the bibliography is contained in the TeX source. Three passes are recommended for a clean build so that theorem/citation pagination and all internal cross-references stabilize. No BibTeX or Biber pass is required.
