FROM debian:latest

RUN apt-get update -y
RUN apt-get install texlive-latex-recommended -y

ENTRYPOINT [ "/bin/bash" ]
