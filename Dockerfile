# python base image in the container from Docker Hub
FROM python:3.12-slim

# install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# copy files to the /app folder in the container
ADD routes /app/routes
ADD utils /app/utils
ADD agents /app/agents
ADD config /app/config
COPY ./main.py /app/main.py
COPY ./pyproject.toml /app/pyproject.toml
COPY ./uv.lock /app/uv.lock
COPY ./README.md /app/README.md

# set the working directory in the container to be /app
WORKDIR /app

# install required packages
RUN uv sync --frozen --no-dev
RUN uv run python -m spacy download en_core_web_sm

# expose the port that uvicorn will run the app on
ENV PORT=8000
EXPOSE 8000

# execute the command python main.py (in the WORKDIR) to start the app
CMD ["uv", "run", "python", "main.py"]
