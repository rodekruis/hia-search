# hia-search

Search engine for [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API to search [HIA](https://github.com/rodekruis/helpful-information).

Based on [langchain](https://github.com/langchain-ai/langchain), powered by OpenAI models..

Uses [Poetry](https://python-poetry.org/) for dependency management.

## API Usage

See [the docs](https://hia-chatbot.azurewebsites.net/docs).

### Configuration

```sh
cp example.env .env
```

Edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

```sh
poetry install
uvicorn main:app --reload
```

### Run with Docker

```sh
docker compose up --detach
```

