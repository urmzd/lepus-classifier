FROM debian:latest

RUN apt-get update -y
RUN apt-get install texlive-latex-extra -y

ENTRYPOINT [ "/bin/bash" ]
