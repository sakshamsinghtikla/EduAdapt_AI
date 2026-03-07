from pydantic import BaseModel


class Settings(BaseModel):
    target_probability: float = 0.65
    concept_mastery_min: float = 0.05
    concept_mastery_max: float = 0.95
    history_window: int = 20


settings = Settings()
