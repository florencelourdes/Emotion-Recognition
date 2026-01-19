import logging
import sys

LOG_NAME = "./model.log"
logger = logging.getLogger()


def init_logger():
    logger.info("Starting logger")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_NAME)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
