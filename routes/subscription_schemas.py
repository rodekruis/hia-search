from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, HttpUrl, StringConstraints, field_validator
from pydantic_core import PydanticCustomError

SheetId = Annotated[
    str,
    StringConstraints(
        strict=True, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"
    ),
]


class InstancePayload(BaseModel):
    instance: HttpUrl

    @field_validator("instance", mode="before")
    @classmethod
    def validate_instance_input(cls, value: object) -> str:
        if not isinstance(value, str):
            raise PydanticCustomError("instance_string", "instance must be a string")
        if any(character.isspace() for character in value):
            raise ValueError("instance must not contain whitespace")
        if (
            not value.lower().startswith("https://")
            or not urlsplit(value).netloc
            or "@" in urlsplit(value).netloc
        ):
            raise ValueError("instance must be an absolute HTTPS URL")
        return value

    @field_validator("instance")
    @classmethod
    def validate_instance_url(cls, value: HttpUrl) -> HttpUrl:
        if (
            value.scheme != "https"
            or value.username is not None
            or value.password is not None
        ):
            raise PydanticCustomError(
                "https_instance", "instance must use HTTPS without credentials"
            )
        return value

    @property
    def normalized_instance(self) -> str:
        # Validate the full input before stripping; do not revalidate the encoded output.
        return str(self.instance).split("?", 1)[0].split("#", 1)[0]


class SubscriptionPayload(InstancePayload):
    googleSheetIds: list[SheetId] = Field(strict=True)


class UnsubscribePayload(InstancePayload):
    googleSheetId: SheetId


class SubscriptionResponse(BaseModel):
    instance: str
    googleSheetIds: list[str]


class UnsubscribeResponse(BaseModel):
    instance: str
    googleSheetId: str
