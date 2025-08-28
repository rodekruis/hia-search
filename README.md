# `hia-search`

Search engine for [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [`dockerized`](https://www.docker.com/) [python](https://www.python.org/) API to search [HIA](https://github.com/rodekruis/helpful-information).

Based on [`langchain`](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Largely inspired by the projects [`knowledge-enriched-chatbot`](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [`hia-search-engine`](https://github.com/rodekruis/hia-search-engine), kudos to the authors.

## API Usage

## `/create-vector-store`

The `/create-vector-store` endpoint accepts a `googleSheetId` parameter, fetches all data from the Q&A sheet, and creates an [index](https://learn.microsoft.com/en-us/azure/search/search-what-is-an-index) in Azure AI Search. If the index already exists, its content will be updated.

If the Q&A sheet is not publicly accessible, you can pass its content to the `data` parameter. The content must be a valid JSON object structured as [this](https://github.com/rodekruis/helpful-information/blob/main/data/test-sheet-id-1/values/Q%26As.json).

## `/search`

The `/search` endpoint accepts three parameters:
* `query`: the search query
* `googleSheetId`: the Google Sheet ID (must be already indexed via `/create-vector-store`)
* `lang`: the language of the search query; results will be translated to this language
* `k`: the number of results to return

and returns a list of relevant questions and answers, in this format:

```json
 [
    {
        "categoryID": 8,
        "subcategoryID": 38,
        "slug": "disabilities",
        "question": "Where can I go when I have special care needs?",
        "answer": "To receive specialist care, you often first need a referral from your General Practitioner (GP).",
        "score": 0.3011241257190705,
        "children": [
            {
                "categoryID": 8,
                "subcategoryID": 38,
                "question": "Social Security Act (WMO)",
                "answer": "Support for persons with disabilities is provided through the Social Security Act (WMO).",
                "score": 0
            }
        ]
    }
]
 ```

For the rest, see [the docs](https://hia-chatbot.azurewebsites.net/docs).

## Configuration

```sh
cp example.env .env
```

Edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

```sh
pip install poetry
poetry install --no-root
uvicorn main:app --reload
```

### Run with Docker

```sh
docker build -t hia-search .
```

