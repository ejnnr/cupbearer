FROM ubuntu:22.04

# Expose general port
EXPOSE 3000
# Expose port for jupyter
EXPOSE 8888
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# tzdata is installed as a dependency and asks interactive questions otherwise:
ARG DEBIAN_FRONTEND=noninteractive

# Install python 3.10
RUN apt-get update \
    && apt install -y python3 python3-pip curl git vim \
    && rm -rf /var/lib/apt/lists/*

COPY . /code/abstractions
WORKDIR /code/abstractions
RUN ./install.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
