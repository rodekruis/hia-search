# `hia-search`

Search and chat with [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [Python](https://www.python.org/) API to search and chat with [HIA](https://github.com/rodekruis/helpful-information).

Based on [LangChain](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Largely inspired by the projects [`knowledge-enriched-chatbot`](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [`hia-search-engine`](https://github.com/rodekruis/hia-search-engine), kudos to the authors.

## Usage

### 1. Prepare the data

Both the chat and search service need HIA content to be transformed into _embeddings_, i.e. numerical representations of text that capture semantic meaning in a high-dimensional vector space. Embeddings are stored in dedicated databases called _vector stores_.

The first step is then to generate the embeddings of your specific HIA instance using the  `/create-vector-store` endpoint, which you can call directly from [the swagger UI](https://hia-search.azurewebsites.net). You will need to authenticate with `API_KEY_WRITE`, see Bitwarden.

>[!NOTE]
> Currently, each search or chat service is linked to only one Google Sheets file, a.k.a. _region_ in the HIA terminology. If it is not clear what this means, see [how HIA works](https://github.com/rodekruis/helpful-information?tab=readme-ov-file#how-it-works).

The `/create-vector-store` endpoint accepts a `googleSheetId` body parameter, fetches all data from the Q&A sheet, and creates an [index](https://learn.microsoft.com/en-us/azure/search/search-what-is-an-index) in Azure AI Search, Azure's native vector database. If the index already exists, its content will be updated.

If the Q&A sheet is not publicly accessible, you can pass its content under the `data` body parameter. The content must be a valid JSON object structured as the [test-data from the `helpful-information`-app](https://github.com/rodekruis/helpful-information/blob/main/data/test-sheet-id-1/values/Q%26As.json).

### 2. Set up chat service

The chat service is based on [Twilio incoming Messaging Webhooks](https://www.twilio.com/docs/usage/webhooks/messaging-webhooks#incoming-message-webhook). You need an active Twilio account and an active phone or WhatsApp number, see [how to buy one](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console).

Select the active number in Twilio and [configure the webhook](https://www.twilio.com/docs/messaging/tutorials/how-to-receive-and-reply/python#configure-your-webhook-url) using the `/chat-twilio-webhook` endpoint, with a `googleSheetId` as query parameter. Example:
```
https://hia-search.azurewebsites.net/chat-twilio-webhook?googleSheetId=14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04
```
The answer will be sent via message directly to the user. The phone number of the user will be used as thread ID, for the chat model to remember the conversation.

### 3. Set up search service

To search HIA content, use the `/search` endpoint. It accepts three parameters:
* `query`: the search query
* `googleSheetId`: the Google Sheet ID
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

 This endpoint is protected with the `API_KEY` environment variable. As this key will be stored by the client-application in plain-text, visible in the browser, it should be considered public. Its main purpose is to prevent abuse of the API by unauthorized users (with possible future measures against it).

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

