"""Reading the Chat sheet: the gviz CSV header, labelled keys and support contacts."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd


@patch("utils.prompt_loader.pd.read_csv")
def test_requests_single_header_row(mock_read_csv):
    """gviz must not guess the header size: it merges the first data row into the header."""
    from utils.prompt_loader import PromptLoader

    mock_read_csv.return_value = pd.DataFrame({"#KEY": ["#system-prompt"], "#VALUE": ["Hi"]})
    PromptLoader(document_type="googlesheet", document_id="sheet123").get_values()

    url = mock_read_csv.call_args.args[0]
    assert "/spreadsheets/d/sheet123/gviz/tq?" in url
    assert "headers=1" in url
    assert "sheet=Chat" in url


@patch("utils.prompt_loader.pd.read_csv")
def test_warns_when_header_merged_with_first_row(mock_read_csv, caplog):
    """What gviz returns without headers=1: the first row is glued onto the header."""
    from utils.prompt_loader import PromptLoader

    mock_read_csv.return_value = pd.DataFrame(
        {
            "#KEY Main System Prompt\n#system-prompt": ["Emergency Contact\n#support-emergency"],
            "#VALUE # Role\nYou are Aidly.": ["112"],
        }
    )
    with caplog.at_level(logging.WARNING, logger="utils.prompt_loader"):
        values = PromptLoader(document_type="googlesheet", document_id="x").get_values()

    assert values == {}
    assert "no #KEY/#VALUE columns" in caplog.text


@patch("utils.prompt_loader.pd.read_csv")
def test_reads_support_contacts_with_labelled_keys(mock_read_csv):
    """Key cells carry a label before the tag, as in the HIA Chat sheet template."""
    from utils.prompt_loader import PromptLoader, clear_prompt_cache, get_support_contacts

    mock_read_csv.return_value = pd.DataFrame(
        {
            "#KEY": [
                "Main System Prompt\n#system-prompt",
                "Emergency Contact\n#support-emergency",
                "SGBV Contact\n#support-gbv",
            ],
            "#VALUE": ["You are Aidly.", "112", "Centrum Seksueel Geweld: 0800-0188"],
        }
    )
    values = PromptLoader(document_type="googlesheet", document_id="x").get_values()
    assert values == {
        "#system-prompt": "You are Aidly.",
        "#support-emergency": "112",
        "#support-gbv": "Centrum Seksueel Geweld: 0800-0188",
    }

    clear_prompt_cache()
    try:
        assert get_support_contacts("labelled-keys-sheet") == {
            "emergency": "112",
            "gbv": "Centrum Seksueel Geweld: 0800-0188",
        }
    finally:
        clear_prompt_cache()
