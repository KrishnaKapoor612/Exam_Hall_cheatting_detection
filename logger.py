import logging
from config import LOG_FILE

def setup_logger():
    """For logging the result in log file"""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("MalpracticeLogger")
    logger.info("=== Session Started - Malpractice Detection Log ===")
    return logger