import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE_PREPROCESS = LOG_DIR / "preprocess.log"
LOG_FILE_TRAIN_MODEL = LOG_DIR / "train_model.log"
LOG_FILE_EVALUATE_MODEL = LOG_DIR / "test.log"
HELPER_FILE_EVALUATE=LOG_DIR/"helper.log"
APP_FILE=LOG_DIR/"app.log"

def get_logger(name: str, log_file: Path, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

preprocess_logger = get_logger("preprocess", LOG_FILE_PREPROCESS)
train_logger = get_logger("train_model", LOG_FILE_TRAIN_MODEL)
evaluate_logger = get_logger("test", LOG_FILE_EVALUATE_MODEL)
helper_logger=get_logger("helper.log",HELPER_FILE_EVALUATE)
app_logger=get_logger("app.log",APP_FILE)

def log_info(message, logger=None):
    if logger:
        logger.info(message)
    else:
        logging.info(message)

def log_error(message, logger=None):
    if logger:
        logger.error(message)
    else:
        logging.error(message)

def log_warning(message, logger=None):
    if logger:
        logger.warning(message)
    else:
        logging.warning(message)
