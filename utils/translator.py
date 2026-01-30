import requests
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def translate(from_lang: str, to_lang: str, text: str) -> str:
    """Translate text from one language to another."""
    if pd.isna(text) or text.strip() == "":
        return ""
    translator_params = {"api-version": "3.0", "to": [to_lang], "from": [from_lang]}
    translator_headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("MSCOGNITIVE_KEY"),
        "Ocp-Apim-Subscription-Region": os.getenv("MSCOGNITIVE_LOCATION"),
        "Content-type": "application/json",
    }
    translator_url = "https://api.cognitive.microsofttranslator.com/translate"
    translator_response = requests.post(
        translator_url,
        params=translator_params,
        headers=translator_headers,
        json=[{"text": text}],
    ).json()
    return translator_response[0]["translations"][0]["text"]


def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    if pd.isna(text) or text.strip() == "":
        return "en"
    detector_params = {"api-version": "3.0"}
    detector_headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("MSCOGNITIVE_KEY"),
        "Ocp-Apim-Subscription-Region": os.getenv("MSCOGNITIVE_LOCATION"),
        "Content-type": "application/json",
    }
    detector_url = "https://api.cognitive.microsofttranslator.com/detect"
    detector_response = requests.post(
        detector_url,
        params=detector_params,
        headers=detector_headers,
        json=[{"text": text}],
    ).json()
    return detector_response[0]["language"]
