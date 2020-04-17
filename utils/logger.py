import logging, os
from logging.handlers import RotatingFileHandler

def get_logger(name):
    """
    Args:
        name(str):생성할 log 파일명입니다.

    Returns:
         생성된 logger객체를 반환합니다.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if not os.path.exists(os.path.join("..", "logs")):
        os.makedirs(os.path.join("..", "logs"))

    rotate_handler = RotatingFileHandler(
        filename=os.path.join("..", "logs", name + ".log"),
        mode='a',
        maxBytes=1024 * 1024 * 10,
        backupCount=5
    )
    formatter = logging.Formatter(
        '[%(levelname)s]-%(asctime)s-%(filename)s:%(lineno)s:%(message)s',
        # datefmt="%Y-%m-%d %H:%M:%S"
        datefmt="%H:%M:%S"
    )
    rotate_handler.setFormatter(formatter)
    logger.addHandler(rotate_handler)
    return logger