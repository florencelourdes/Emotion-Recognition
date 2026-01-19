import os
from models import B0Model, B3Model, Resnet50Model
import kagglehub
from train import run_emotion_recognition_model
from logger import init_logger
import logging
from util import copy_all
from settings import settings

init_logger()
logger = logging.getLogger()

datasetpath = kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")

copy_all(os.path.join(datasetpath, "processed_data"), settings.DATASET_PATH)

batch_sizes = [16, 32, 64]
num_epochs = 30
for batch_size in batch_sizes:

    logger.info(f"Running B0Model: epochs={num_epochs}, batch={batch_size}")
    run_emotion_recognition_model(B0Model, num_epochs, batch_size)
    logger.info(f"B0Model Complete")

    logger.info(f"Running B3Model: epochs={num_epochs}, batch={batch_size}")
    run_emotion_recognition_model(B3Model, num_epochs, batch_size)
    logger.info(f"B3Model Complete")

    logger.info(f"Running Resnet50Model: epochs={num_epochs}, batch={batch_size}")
    run_emotion_recognition_model(Resnet50Model, num_epochs, batch_size)
    logger.info(f"Resnet50Model Complete")
