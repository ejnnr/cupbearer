FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

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

# Install python 3.11
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y python3.11 curl git vim \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3.11 -
# Add poetry to path
ENV PATH="/root/.local/bin:$PATH"

# Copy only necessary dependencies to build virtual environment.
# This minimizes how often this layer needs to be rebuilt.
COPY poetry.lock pyproject.toml /code/abstractions/
WORKDIR /code/abstractions
# Without this, poetry will try to use the system python (3.10)
RUN poetry env use python3.11
RUN poetry install --no-interaction --no-ansi && poetry cache clear pypi --all
# clear the directory again (this is necessary so that CircleCI can checkout
# into the directory)
RUN rm poetry.lock pyproject.toml

ENTRYPOINT ["/usr/bin/tini", "--"]