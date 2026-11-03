from psycopg.conninfo import make_conninfo
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SubscriptionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SUBSCRIPTION_DB_", extra="ignore")

    host: str
    port: int = Field(default=5432, ge=1, le=65535)
    name: str
    user: str
    password: SecretStr
    sslmode: str = "verify-full"
    pool_min_size: int = Field(default=1, ge=1)
    pool_max_size: int = Field(default=5, ge=1)
    timeout: float = Field(default=10, gt=0)

    @model_validator(mode="after")
    def validate_pool_sizes(self):
        if self.pool_max_size < self.pool_min_size:
            raise ValueError("pool_max_size must be at least pool_min_size")
        return self

    def conninfo(self) -> str:
        return make_conninfo(
            host=self.host,
            port=self.port,
            dbname=self.name,
            user=self.user,
            password=self.password.get_secret_value(),
            sslmode=self.sslmode,
            connect_timeout=max(1, int(self.timeout)),
            application_name="hia-subscriptions",
        )
