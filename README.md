# hia-search

Search engine for [HIA](https://github.com/rodekruis/helpful-information).

## Description

Synopsis: a [dockerized](https://www.docker.com/) [python](https://www.python.org/) API to search [HIA](https://github.com/rodekruis/helpful-information).

Based on [langchain](https://github.com/langchain-ai/langchain), powered by language models. Uses [Poetry](https://python-poetry.org/) for dependency management and [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) for indexing and searching.

Largely inspired by [this](https://github.com/deloitte-nl/knowledge-enriched-chatbot) and [this](https://github.com/rodekruis/hia-search-engine) project, kudos to the authors.

## API Usage

## `/create-vector-store`

The `/create-vector-store` endpoint accepts a `googleSheetId` parameter, fetches all data from the Q&A sheet, and creates an [index](https://learn.microsoft.com/en-us/azure/search/search-what-is-an-index) in Azure AI Search. If the index already exists, its content will be updated.


## `/search`

The `/search` endpoint accepts three parameters:
* `query`: the search query
* `googleSheetId`: the Google Sheet id (must be already indexed via `/create-vector-store`)
* `k`: the number of results to return

and returns a list of relevant questions and answers, in this format:

```json
 [
    {
        "category": "Health & Wellbeing",
        "subcategory": "Cancer",
        "slug": null,
        "question": "I have cancer and need treatment, where can I go? ",
        "answer": "All hospitals in the Netherlands open their doors to provide medical care to Ukrainian refugees. This also applies to people from Ukraine with cancer. They can continue their treatment in the Netherlands. Of course in consultation with a doctor.\n\nFirst, you have to go to a general practitioner.",
        "score": 0.3011241257190705,
        "children": [
            {
                "content": "Support for persons with disabilities is provided through the Social Security Act (Wmo). You can also make use of more long-term care, such as care from the Wmo. This care is provided by municipalities. The WMO regulates help and support for citizens, so that they can live independently at home for as long as possible and continue to participate in society. ",
                "answer": "Social Security Act (WMO)",
                "score": 0
            },
            {
                "content": "Municipalities and MEE, an association of 20 social service organizations, help people with special needs. You will be assigned a social worker who will guide not only you or your child, but the whole family. \n\n<b>Please note:</b> not all help is free. Some aid (for example orthopedic shoes or hearing aids) must be partially paid for yourself. \n\nFor most recent information see: https://www.mee.nl/",
                "answer": "MEE",
                "score": 0.2600285887718202
            }
        ]
    },
    {
        "category": "Health & Wellbeing",
        "subcategory": "Specialized care & disabilities",
        "slug": null,
        "question": "I want advice about sexual health / STI / contraception / pregnancy, where can I go?",
        "answer": "This depends on the municipality where you are located. Check the information listed below.",
        "score": 0.0,
        "children": [
            {
                "content": "You can talk to the GGD about every topic concerning your body, your relations and sex. They can provide care regarding your sexual health, sexual orientation, gender issues,  sexwork or sexual violence. Come in to talk about ways to protect yourself with  PEP/PrEP, contraception and vaccinations. This is a LHBT-friendly centre. You don't need  ID / BSN / health-insurance.\n\nYou have to call or use the website link to make an appointment. \n\nEveryone is welcome, the care is free of charge. You do not need a referral by a doctor. \n\nAll healthcare workers have professional secrecy.\n\nhttps://afspraak.ggdaphrodite.nl/?lang=en\n \nPhone: (020) 555 5822\n\nAddress:\nGGD Amsterdam Centrum voor Seksuele Gezondheid\nNieuwe Achtergracht 100\n1018 WT Amsterdam\n(near Weesperplein metro station)\n",
                "question": "GGD Center for Sexual Health in Amsterdam",
                "score": 0.3008084032535553
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

