FROM python:3.12-slim

ARG NAME # Ensure this is passed during 'oc new-build --build-arg NAME=presidio-analyzer'
ARG NLP_CONF_FILE=presidio_analyzer/conf/default.yaml
ARG ANALYZER_CONF_FILE=presidio_analyzer/conf/default_analyzer.yaml
ARG RECOGNIZER_REGISTRY_CONF_FILE=presidio_analyzer/conf/default_recognizers.yaml

ENV PIP_NO_CACHE_DIR=1

# --- Poetry Configuration ---
# 1. Create virtualenv in the project directory (.venv).
# 2. Prevent Poetry from trying to create a venv at runtime if it thinks one is missing.
# 3. Disable interactive prompts.
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
# --- End Poetry Configuration ---

ENV ANALYZER_CONF_FILE=${ANALYZER_CONF_FILE}
ENV RECOGNIZER_REGISTRY_CONF_FILE=${RECOGNIZER_REGISTRY_CONF_FILE}
ENV NLP_CONF_FILE=${NLP_CONF_FILE}

ENV PORT=3000
ENV WORKERS=1

# Ensure NAME is set, otherwise WORKDIR and COPY paths will be incorrect
RUN if [ -z "$NAME" ]; then echo "Error: Build argument NAME is not set. It's used for WORKDIR and COPY paths." && exit 1; fi \
    && mkdir -p /usr/bin/${NAME}

WORKDIR /usr/bin/${NAME}

# Copy configuration files first if they might be needed by other steps,
# or if their locations are fixed relative to WORKDIR.
# Ensure the source paths for these COPY commands are correct in your build context.
# Example: if ANALYZER_CONF_FILE is "presidio_analyzer/conf/default_analyzer.yaml",
# the source file should be at "presidio_analyzer/conf/default_analyzer.yaml" relative to your Dockerfile.
COPY ${ANALYZER_CONF_FILE} /usr/bin/${NAME}/${ANALYZER_CONF_FILE}
COPY ${RECOGNIZER_REGISTRY_CONF_FILE} /usr/bin/${NAME}/${RECOGNIZER_REGISTRY_CONF_FILE}
COPY ${NLP_CONF_FILE} /usr/bin/${NAME}/${NLP_CONF_FILE}

# Install essential build tools
RUN apt-get update \
    && apt-get install -y build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry itself. Using curl is a common method.
# Consider pinning the Poetry version.
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy project and lock files
COPY ./pyproject.toml ./poetry.lock* /usr/bin/${NAME}/

# Install dependencies using Poetry.
# This will create .venv in /usr/bin/${NAME}/.venv due to POETRY_VIRTUALENVS_IN_PROJECT=true
RUN poetry install --no-root --only=main -E server

# install nlp models specified in NLP_CONF_FILE
COPY ./install_nlp_models.py /usr/bin/${NAME}/
# Ensure the NLP_CONF_FILE is accessible for the script; it was copied earlier.
RUN poetry run python install_nlp_models.py --conf_file ${NLP_CONF_FILE}

# Copy the rest of your application code
COPY . /usr/bin/${NAME}/

EXPOSE ${PORT}
CMD poetry run gunicorn -w $WORKERS -b 0.0.0.0:$PORT 'app:create_app()'