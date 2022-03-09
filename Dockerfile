FROM debian:latest

# Set up dependencies.
# Build documentation.
RUN apt update
RUN apt-get install texlive

COPY . /

# Build program.

# TODO
ENTRYPOINT [ "/bin/bash" ]
