# Using cuda11 because that's what the v470 drivers support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

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

# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY requirements.txt cuda-requirements.txt torch-requirements.txt install.sh /code/abstractions/
WORKDIR /code/abstractions
RUN ./install.sh
# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm ./*

ENTRYPOINT ["/usr/bin/tini", "--"]