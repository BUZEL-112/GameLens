"""
API configuration via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    artifacts_path: str = "model_artifacts"
    n_candidates: int = 100
    max_genres_per_response: int = 3


settings = Settings()
