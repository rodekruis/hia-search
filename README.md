# hia-search

Search engine for [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API to search [HIA](https://github.com/rodekruis/helpful-information).

Based on [langchain](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Largely inspired by [this](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [this](https://github.com/rodekruis/hia-search-engine) project, kudos to the authors.

## API Usage

See [the docs](https://hia-chatbot.azurewebsites.net/docs).

### Configuration

```sh
cp example.env .env
```

and edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

```sh
pip install poetry
poetry install --no-root
uvicorn main:app --reload
```

### Run with Docker

```sh
docker compose up --detach
```

