import datetime
import logging
import os


def mylog():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logfile = "./log/" + str(datetime.datetime.now().month) + "-" + str(datetime.datetime.now().day) + "-" + str(
        datetime.datetime.now().hour) + "-" + str(datetime.datetime.now().minute) + \
              os.path.split(__file__)[-1].split(".")[0] + '.log'
    fileHandler = logging.FileHandler(logfile, mode='w', encoding='UTF-8')
    fileHandler.setLevel(logging.NOTSET)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger
