# `hia-search`

Search and chat with [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [Python](https://www.python.org/) API to search and chat with [HIA](https://github.com/rodekruis/helpful-information).

Based on [LangChain](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Inspired by [`knowledge-enriched-chatbot`](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [`HIA-search-engine`](https://github.com/PetrovViktor/HIA-search-engine), kudos to the authors.

## Usage

### 1. Set up HIA

[Set up a HIA instance](https://github.com/rodekruis/helpful-information/blob/main/docs/Guide-How_to_set_up_an_instance.md) and populate its content.

### 2. Set up the search service

  - Find the item called `HIA Search API-key(s) [production]` in Bitwarden and copy the value of `API_KEY`
  - Go to the HIA repository's "**Settings**" > "**Secret and variables**" > "**Actions**" > "**New repository secret**"
  - Name the secret: `SEARCH_API_KEY`and insert the API Key as value
  - In the HIA repository, go to the file: `.github/workflows/deploy-github-pages.yml` and click "**Edit this file**" (pencil icon)
  - Under `jobs` > `deploy` > `steps` > `Build` > `env` add the two variables `SEARCH_API` and `SEARCH_API_KEY` as follows

  ```yaml
        - name: Build
        working-directory: 'helpful-information'
        # NOTE: When the instance will be run on a custom (sub-)domain, remove the `--base-href`-flag+value.
        run: 'npm run build:production -- --output-path=../www --base-href=/${GITHUB_REPOSITORY#*/}/'
        env:
          # See all variables: https://github.com/rodekruis/helpful-information/blob/main/.env.example
          NG_PRODUCTION: 'true'
          GOOGLE_SHEETS_API_KEY: ${{ secrets.GOOGLE_SHEETS_API_KEY }}
          GOOGLE_SHEETS_API_URL: 'https://sheets.googleapis.com/v4/spreadsheets'
          SEARCH_API: 'https://hia-search.azurewebsites.net/search'
          SEARCH_API_KEY: ${{ secrets.SEARCH_API_KEY }}
  ```
  - Redeploy HIA by triggering the deployment workflow: `Actions` > `Deploy to GitHub Pages` > `Run workflow`
  

### 2. Set up the chat service

The chat service is based on [Twilio incoming Messaging Webhooks](https://www.twilio.com/docs/usage/webhooks/messaging-webhooks#incoming-message-webhook). You need an active Twilio account and an active phone or WhatsApp number, see [how to buy one](https://help.twilio.com/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console).

Select the active number in Twilio and [configure the webhook](https://www.twilio.com/docs/messaging/tutorials/how-to-receive-and-reply/python#configure-your-webhook-url) using the `/chat-twilio-webhook` endpoint, with a `googleSheetId` as query parameter. Example:
```
https://hia-search.azurewebsites.net/chat-twilio-webhook?googleSheetId=14NZwDa8DNmH1q2Rxt-ojP9MZhJ-2GlOIyN8RF19iF04
```
The answer will be sent via message directly to the user. The phone number of the user will be used as thread ID, for the chat model to remember the conversation.

>[!NOTE]
>The instructions that the chatbot will follow are by default [these ones](config/rag_agent_prompt.txt). If you want to customize them, create a new sheet named `Chat` in your HIA Google Sheet file following [this template](https://docs.google.com/spreadsheets/d/1op6Ouyxtwv4f8GAEAMSn5PVzcXtfZuftMiLYWsX0pbs/edit?pli=1&gid=1707339525#gid=1707339525), then insert the desired instructions under `#VALUE`, cell `B2`. Make sure to follow [best practices in prompt engineering](https://www.promptingguide.ai/introduction/tips); if it's the first time you do this, make sure the CEA Data Specialist reviews what you wrote.


### 3. Keep your data up to date

Both the chat and search services need HIA content to be transformed into _embeddings_, i.e. numerical representations of text that capture semantic meaning in a high-dimensional vector space. Embeddings are stored in dedicated databases called _vector stores_. When searching or chatting with HIA, the user query 

Generating and storing the embeddings of your specific HIA instance is done automatically the first time you call the chat or search endpoints, if the HIA Google Sheet file is publicly accessible. If the Google Sheet file is not publicly accessible or you need to update it, you can use the  `/create-vector-store` endpoint, which you can call directly from [the swagger UI](https://hia-search.azurewebsites.net). You will need to authenticate with `API_KEY_WRITE`, which you find in Bitwarden. See API reference below.

>[!NOTE]
> Currently, each search or chat service is linked to only one Google Sheets file, a.k.a. _region_ in the HIA terminology. If it is not clear what this means, see [how HIA works](https://github.com/rodekruis/helpful-information?tab=readme-ov-file#how-it-works).

## API Reference

If not here, see the [`/docs`](https://hia-search.azurewebsites.net/docs).

### `/create-vector-store`

The `/create-vector-store` endpoint accepts a `googleSheetId` body parameter, fetches all data from the Q&A sheet, and creates an [index](https://learn.microsoft.com/en-us/azure/search/search-what-is-an-index) in Azure AI Search, Azure's native vector database. If the index already exists, its content will be updated.

If the Q&A sheet is not publicly accessible, you can pass its content under the `data` body parameter. The content must be a valid JSON object structured as the [test-data from the `helpful-information`-app](https://github.com/rodekruis/helpful-information/blob/main/data/test-sheet-id-1/values/Q%26As.json).

🔐 This endpoint is protected with the API_KEY_WRITE environment-variable, to prevent unauthorized users from modifying the index.

### `/search`

The `/search` endpoint accepts three parameters:
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

