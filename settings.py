from typing import Dict, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Settings can be set in .env file
    DATASET_PATH: str = "./images"
    LABELS: List[str] = ["angry", "sad", "disgust", "fear", "happy", "neutral", "surprise"]
    SATISFACTION_LABELS: List[str] = ["dissatisfied", "neutral", "satisfied"]
    EMOTION_TO_SATISFACTION: Dict[str, str] = {
        "angry": "dissatisfied",
        "sad": "dissatisfied",
        "disgust": "dissatisfied",
        "fear": "dissatisfied",
        "happy": "satisfied",
        "neutral": "neutral",
        "surprise": "satisfied",
    }

    class Config:
        env_file = ".env"


settings = Settings()
