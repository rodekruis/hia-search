import urllib
import pandas as pd
from utils.logger import logger
from fastapi import HTTPException


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
        except (IndexError, KeyError) as e:
            prompt = ""
        return prompt
