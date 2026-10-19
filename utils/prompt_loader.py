import functools
import logging
import os
import time
import urllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "rag_agent_prompt.txt"
DIRECT_ANSWER_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "direct_answer_prompt.txt"
LANGUAGE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "language_prompt.txt"
WELLBEING_PROMPT_PATH = Path(__file__).resolve().parent.parent / "config" / "wellbeing_prompt.txt"

# Support contacts per instance: Chat sheet row `#support-<need>`, one contact each
SUPPORT_NEEDS = {
    "emergency": "Emergency services",
    "psychosocial_support": "Psychosocial support",
    "talk_to_staff": "Talk to a staff member",
    "child_protection": "Child protection",
    "gbv": "Sexual or gender-based violence",
}

# Cache the Chat sheet settings to reduce latency.
PROMPT_CACHE_TTL_S = float(os.environ.get("PROMPT_CACHE_TTL_S", "300"))


@dataclass(frozen=True)
class ChatSettings:
    system_prompt: str
    support_contacts: dict[str, str]


_prompt_cache: dict[str, tuple[float, ChatSettings]] = {}


def _chat_settings(google_sheet_id: str) -> ChatSettings:
    now = time.monotonic()
    cached = _prompt_cache.get(google_sheet_id)
    if cached is not None and now - cached[0] < PROMPT_CACHE_TTL_S:
        return cached[1]

    loader = PromptLoader(document_type="googlesheet", document_id=google_sheet_id)
    values = loader.get_values()
    prompt = loader.get_prompt(values) or DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
    contacts = {
        need: values[f"#support-{need}"]
        for need in SUPPORT_NEEDS
        if values.get(f"#support-{need}")
    }
    settings = ChatSettings(system_prompt=prompt, support_contacts=contacts)
    _prompt_cache[google_sheet_id] = (now, settings)
    return settings


def get_system_prompt(google_sheet_id: str) -> str:
    """System prompt for a HIA instance: its sheet's `#system-prompt`, else the default file."""
    return _chat_settings(google_sheet_id).system_prompt


def get_support_contacts(google_sheet_id: str) -> dict[str, str]:
    """Support contacts of a HIA instance by need; needs without a contact are left out."""
    return _chat_settings(google_sheet_id).support_contacts


def clear_prompt_cache() -> None:
    _prompt_cache.clear()


@functools.cache
def get_direct_answer_prompt() -> str:
    """Instance-agnostic rules for turns answered without retrieval (greetings, small talk, off-topic)."""
    return DIRECT_ANSWER_PROMPT_PATH.read_text(encoding="utf-8").strip()


@functools.cache
def get_language_prompt() -> str:
    """Instance-agnostic rules to reply in the user's language; chat messages are not machine-translated."""
    return LANGUAGE_PROMPT_PATH.read_text(encoding="utf-8").strip()


@functools.cache
def _wellbeing_rules() -> str:
    return WELLBEING_PROMPT_PATH.read_text(encoding="utf-8").strip()


def get_wellbeing_prompt(support_contacts: dict[str, str]) -> str:
    """Instance-agnostic rules for users in distress, followed by the instance's support contacts."""
    lines = [
        f"- {label}: {support_contacts[need]}"
        for need, label in SUPPORT_NEEDS.items()
        if support_contacts.get(need)
    ]
    if not support_contacts.get("emergency"):
        lines.insert(
            0,
            f"- {SUPPORT_NEEDS['emergency']}: no contact listed; tell them to contact "
            "their local emergency services.",
        )
    return f"{_wellbeing_rules()}\n\n# Support contacts\n" + "\n".join(lines)


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

    def get_values(self) -> dict[str, str]:
        """All `#KEY` -> `#VALUE` rows of the Chat sheet (empty when it is missing or malformed)."""
        if self.document_type.lower() == "googlesheet":
            sheet_name = "Chat"
            url = f"https://docs.google.com/spreadsheets/d/{self.document_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            try:
                df = pd.read_csv(url)
            except urllib.error.HTTPError as e:
                return {}

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
        try:
            df = df.dropna(subset=["#KEY"])
            return {
                _tag(key): str(value).strip()
                for key, value in zip(df["#KEY"], df["#VALUE"])
                if not pd.isna(value) and str(value).strip()
            }
        except KeyError:
            return {}

    def get_prompt(self, values: dict[str, str] | None = None) -> str:
        """The `#system-prompt` value, "" when there is none."""
        if values is None:
            values = self.get_values()
        return values.get("#system-prompt", "")


def _tag(key: object) -> str:
    """`Main System Prompt\\n#system-prompt` -> `#system-prompt`: HIA key cells put a label before the tag."""
    tags = [part for part in str(key).split() if part.startswith("#")]
    return tags[-1] if tags else str(key).strip()
