import logging
import json
from pathlib import Path
from collections import OrderedDict


def get_logger(name, verbosity=2):
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logger = logging.getLogger(name)
    logger.setLevel(log_levels[verbosity])
    return logger


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)
