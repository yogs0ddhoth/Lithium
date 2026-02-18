from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    api_key: str = "<placeholder_key>"
    api_type: str | None = None
    api_version: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
    )
    anthropic_api_key: str = "<placeholder_key>"
