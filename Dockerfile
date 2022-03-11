FROM debian:latest

RUN apt-get update -y
RUN apt-get install texlive -y

ENTRYPOINT [ "/bin/bash" ]
