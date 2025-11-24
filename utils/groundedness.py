import re
import requests
import os
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()


def detect_groundness(content_text: str, grounding_sources: list, query: str):

    subscription_key = os.environ["AISAFETY_API_KEY"]
    endpoint = os.environ["AISAFETY_ENDPOINT"]
    api_version = os.environ["AISAFETY_API_VERSION"]

    # GPT resources to provide an explanation
    openai_endpoint = os.environ["OPENAI_ENDPOINT"]
    deployment_name = os.environ["MODEL_GROUNDEDNESS"]

    # Build the request body
    data = {
        "domain": "GENERIC",
        "task": "QNA",
        "text": content_text,
        "groundingSources": grounding_sources,
        "reasoning": True,
        "correction": False,
        "qna": {"query": query},
        "llmResource": {
            "resourceType": "AzureOpenAI",
            "azureOpenAIEndpoint": openai_endpoint,
            "azureOpenAIDeploymentName": deployment_name,
        },
    }

    # Send the API request
    url = f"{endpoint}/contentsafety/text:detectGroundedness?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    response = requests.post(url, headers=headers, json=data)

    # Handle the API response
    if response.status_code == 200:
        result = response.json()
        if result["ungroundedDetected"] and result["ungroundedPercentage"] >= 0.25:
            for detail in result["ungroundedDetails"]:
                # Redact ungrounded content
                from_character = detail["offset"]["utf8"]
                to_character = from_character + detail["length"]["utf8"]
                content_text = content_text.replace(
                    content_text[from_character:to_character], ""
                )
                logger.info(
                    f"Ungrounded content redacted: {content_text[from_character:to_character]}. "
                    f"Reason: {detail['reason']}. "
                    f"User query: {query}. "
                    f"Grounding sources: {grounding_sources}."
                )
    else:
        logger.error("Error in detect_groundness:", response.status_code, response.text)
    return content_text
