import os
import time
import urllib
from pathlib import Path

import pandas as pd
from utils.logger import logger
from fastapi import HTTPException

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "rag_agent_prompt.txt"

# The sheet is fetched over HTTP; re-reading it on every chat turn is pure latency.
# Prompt edits in the sheet take effect within this many seconds.
PROMPT_CACHE_TTL_S = float(os.environ.get("PROMPT_CACHE_TTL_S", "300"))
_prompt_cache: dict[str, tuple[float, str]] = {}


def get_system_prompt(google_sheet_id: str) -> str:
    """System prompt for a HIA instance: its sheet's `#system-prompt`, else the default file."""
    now = time.monotonic()
    cached = _prompt_cache.get(google_sheet_id)
    if cached is not None and now - cached[0] < PROMPT_CACHE_TTL_S:
        return cached[1]

    prompt = PromptLoader(document_type="googlesheet", document_id=google_sheet_id).get_prompt()
    if prompt == "":
        prompt = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    _prompt_cache[google_sheet_id] = (now, prompt)
    return prompt


def clear_prompt_cache() -> None:
    _prompt_cache.clear()


class PromptLoader:
    """
    Prompt loading class that:
        1. Loads prompt from source
        2. Return prompt as text
    """

    output: str

    def __init__(
        self,
        document_type: str,
        document_id: str = None,
        document_data: dict = None,
        **kwargs: dict,
    ):
        self.document_type = document_type
        self.document_id = document_id
        self.document_data = document_data
        self.__dict__.update(kwargs)

    def get_prompt(self):
        """
        Loads system-prompt based on the document type. Google Sheet and JSON are currently supported.
        """
        if self.document_type.lower() == "googlesheet":
            sheet_name = "Chat"
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            try:
                df = pd.read_csv(url)
            except urllib.error.HTTPError as e:
                return ""

        elif self.document_type.lower() == "json":
            logger.info("Loading from JSON.")
            if not self.document_data:
                raise HTTPException(
                    status_code=400, detail="No document data provided."
                )
            df = pd.DataFrame(self.document_data["values"])
            new_header = df.iloc[0]  # grab the first row for the header
            df = df[1:]  # take the data less the header row
            df.columns = new_header  # set the header row as the df header
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Loader of document type {self.document_type} not available.",
            )
        # get prompt from column #VALUE and row with #KEY containing #system-prompt
        try:
            df = df.dropna(subset=["#KEY"])
            prompt = (
                df[df["#KEY"].str.contains("#system-prompt")]["#VALUE"]
                .values[0]
                .strip()
            )
        except (IndexError, KeyError, AttributeError) as e:
            prompt = ""
        return prompt
