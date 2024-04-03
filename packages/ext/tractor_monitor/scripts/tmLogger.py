import logging
import logging.handlers

# logging
fileMaxByte = 1024 * 1024 * 1   # 1MB

def tmLogger(name): # name is TrError, TrLive
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(levelname)s|%(name)s] %(asctime)s > %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.handlers.RotatingFileHandler('/var/log/tr_monitor.log', maxBytes=fileMaxByte, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
