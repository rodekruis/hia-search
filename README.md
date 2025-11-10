# `hia-search`

Search engine for [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [`dockerized`](https://www.docker.com/) [python](https://www.python.org/) API to search [HIA](https://github.com/rodekruis/helpful-information).

Based on [`langchain`](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Largely inspired by the projects [`knowledge-enriched-chatbot`](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [`hia-search-engine`](https://github.com/rodekruis/hia-search-engine), kudos to the authors.

## API Usage

## `/create-vector-store`

The `/create-vector-store` endpoint accepts a `googleSheetId` parameter, fetches all data from the Q&A sheet, and creates an [index](https://learn.microsoft.com/en-us/azure/search/search-what-is-an-index) in Azure AI Search. If the index already exists, its content will be updated.

If the Q&A sheet is not publicly accessible, you can pass its content to the `data` parameter. The content must be a valid JSON object structured as the [test-data from the `helpful-information`-app](https://github.com/rodekruis/helpful-information/blob/main/data/test-sheet-id-1/values/Q%26As.json).

🔐 This endpoint is protected with the `API_KEY_WRITE` environment-variable, to prevent unauthorized users from modifying the index.

## `/chat-twilio-webhook`

Chat endpoint for [Twilio Incoming Messaging Webhooks](https://www.twilio.com/docs/usage/webhooks/messaging-webhooks#incoming-message-webhook). To set it up, you first need an active number in Twilio, [here's how to buy one](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console). Select that number and [configure the webhook in Twilio](https://www.twilio.com/docs/messaging/tutorials/how-to-receive-and-reply/python#configure-your-webhook-url) using this endpoint as URL, with the Google sheet ID as parameter (must be already indexed via `/create-vector-store`). Example:
```
https://hia-search-dev.azurewebsites.net/chat-twilio-webhook?google_sheet_id=14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04
```
The chatbot answers will be sent via message to whoever contacted the selected number. The phone number of the user will be used as thread ID (basically, for the model to remember the conversation). 

## `/search`

The `/search` endpoint accepts three parameters:
* `query`: the search query
* `googleSheetId`: the Google Sheet ID (must be already indexed via `/create-vector-store`)
* `lang`: the language of the search query; results will be translated to this language
* `k`: the number of results to return

and returns a list of relevant questions and answers, in this format:

```json
{"results":
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
}
 ```

 🔐 This endpoint is protected with the `API_KEY` environment variable. As this key will be stored by the client-application in plain-text, visible in the browser, it should be considered public. Its main purpose is to prevent abuse of the API by unauthorized users (with possible future measures against it).

For the rest, see [the `/docs`](https://hia-search.azurewebsites.net/docs).  
Or locally at: <http://localhost:8000/docs>.

## Configuration

```sh
cp example.env .env
```

Edit the provided [ENV-variables](./example.env) accordingly.

### Run locally

```sh
pip install poetry
poetry install --no-root
python -m spacy download en_core_web_sm
uvicorn main:app --reload
```

### Run with Docker

```sh
docker build -t hia-search .
```

