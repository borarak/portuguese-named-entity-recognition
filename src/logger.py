import logging
import os
import config

loggers = {}


def custom_logger(name):

    """
    Creates a logger instance
    :param name: name of the logger requested
    :return: an instance of the logger requested
    """

    global loggers

    if loggers.get(name):
        return loggers[name]
    log_file = "portuguese_ner_raktim.log"

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

    fileHandler = logging.FileHandler(os.path.join(config.LOG_DIR, log_file))
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

    log.propagate = False
    loggers[name] = log
    return log