#!/usr/bin/bash

# Save directory where bash is ran
wd=$(pwd) 

function compile-latex {
  cwd=$(dirname "$2")

  cd "$cwd"

  texs=( *.tex )
  _bibs=( *.bib )
  bibs="${_bibs[@]%%.*}"

  # compilation
  for tex in ${texs[@]}; do
    echo $tex
    pdflatex $tex 
  done

  for bib in ${bibs[@]}; do
    bibtex $bib
  done

  for tex in ${texs[@]}; do
    pdflatex $tex
  done

  for tex in ${texs[@]}; do
    pdflatex $tex
  done

  pdfs=( *.pdf )

  for pdf in ${pdfs[@]}; do
    cp $pdf $1
  done
  
  # Go back home
  cd $1
}

# Compile all docs for all tex files
find docs -name '*.tex' | while read file; do compile-latex $wd $file; done
