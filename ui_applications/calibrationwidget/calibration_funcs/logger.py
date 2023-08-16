# Logging
#
# Copyright (c) 2022 山东迈科显微生物科技有限公司
#
# -*- coding: utf-8 -*-

import sys
import logging
import threading
import inspect
import os
from os.path import basename

__all__ = [
    "init_log",
    "debug_log",
    "info_log",
    "warn_log",
    "error_log",
    "fatal_log",
]

_log_level = {
    "DEBUG": ("DEBUG", 10),
    "INFO": ("INFO", 20),
    "WARN": ("WARN", 30),
    "ERROR": ("ERROR", 40),
    "FATAL": ("CRITICAL", 50)
}

_log_helper = threading.local()


def _log(level, msg, *args):
    """
    Log message
    :param level: log level
    :param msg: message
    """
    initialized = getattr(_log_helper, 'initialized', None)
    if initialized is None:
        _log_helper.initialized = False
    elif _log_helper.initialized:
        (frame, filename, lineno, func_name,
         lines, index) = inspect.getouterframes(inspect.currentframe())[2]
        call_info = basename(filename) + "|" + func_name + "|" + str(lineno)
        msg = call_info + " > " + msg
        logger = logging.getLogger(_log_helper.name)
        logger.log(_log_level[level][1], msg, *args)


def init_log(level, filename, name="main"):
    """
    Initialize logger with level, file and thread name. Call this function for each thread before logging.
    :param level: logger level, can be one of DEBUG, INFO, WARN, ERROR, FATAL
    :param filename: logger file name
    :param name: the logger name, can NOT be None or empty
    :return: True if initialize success, otherwise False
    """
    initialized = getattr(_log_helper, 'initialized', None)
    if initialized is None:
        _log_helper.initialized = False

    if not level or not filename or not name:
        print("Invalid argument(s): level[%s], file[%s], name[%s]" %
              (level, filename, name))
        return False
    _log_helper.name = name

    if level not in _log_level:
        print("Invalid level[%s], expect %s" % (level, _log_level))
        return False

    os.makedirs(os.path.dirname(filename), mode=0o755, exist_ok=True)

    try:
        logger = logging.getLogger(_log_helper.name)
        logger.setLevel(_log_level[level][0])

        fmt_str = "%(asctime)s|%(levelname)s|%(process)d|%(threadName)s|" \
                  "%(module)s|%(message)s"
        formatter = logging.Formatter(fmt_str)

        file_handle = logging.FileHandler(filename)
        file_handle.setFormatter(formatter)

        logger.addHandler(file_handle)
    except:
        print("Initialize log failed: catch exception type[%s], value[%s]" %
              (sys.exc_info()[0], sys.exc_info()[1]))
        return False

    _log_helper.initialized = True
    info_log("%s start" % name)
    return True


def debug_log(msg, *args):
    """
    Log debug message
    :param msg: The message
    """
    _log("DEBUG", msg, *args)


def info_log(msg, *args):
    """
    Log information message
    :param msg: The message
    """
    _log("INFO", msg, *args)


def warn_log(msg, *args):
    """
    Log warning message
    :param msg: The message
    """
    _log("WARN", msg, *args)


def error_log(msg, *args):
    """
    Log error message
    :param msg: The message
    """
    _log("ERROR", msg, *args)


def fatal_log(msg, *args):
    """
    Log fatal message
    :param msg: The message
    """
    _log("FATAL", msg, *args)
