from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def create_settings(filename):
    class Settings(BaseSettings):
        log_level: str = Field(default="info", description="Logging level")
        input_file: str = Field(default="arxiv_dataset_all_info.csv",
                                description="File with input dataframe"),
        id_column_name: str = Field(default="id"),
        abstract_column_name: str = Field(default="summary"),
        output_prefix: str = Field(default="arxiv_dataset"),
        model_config = SettingsConfigDict(env_file=filename, env_file_encoding="utf-8")

    return Settings()
