import logging
import requests
import os
from dotenv import load_dotenv
import pandas as pd

logger = logging.getLogger(__name__)

load_dotenv()


class TranslationError(RuntimeError):
    pass


def translate(from_lang: str, to_lang: str, text: str, *, strict: bool = False) -> str:
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
    try:
        response = requests.post(
            translator_url,
            params=translator_params,
            headers=translator_headers,
            json=[{"text": text}],
            timeout=10,
        )
        response.raise_for_status()
        translated = response.json()[0]["translations"][0]["text"]
        if strict and (not isinstance(translated, str) or not translated.strip()):
            raise TranslationError("Translation API returned empty or invalid text.")
        return translated
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        if strict:
            raise TranslationError("Translation API failed.") from None
        logger.error(f"Translation API error: {e}")
        return text


def detect_language(text: str, *, strict: bool = False) -> str:
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
    try:
        response = requests.post(
            detector_url,
            params=detector_params,
            headers=detector_headers,
            json=[{"text": text}],
            timeout=10,
        )
        response.raise_for_status()
        language = response.json()[0]["language"]
        if strict and (not isinstance(language, str) or not language.strip()):
            raise TranslationError("Language detection API returned an invalid language.")
        return language
    except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
        if strict:
            raise TranslationError("Language detection API failed.") from None
        logger.error(f"Language detection API error: {e}")
        return "en"
