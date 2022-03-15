import os
import psutil


def get_memory_usage(process=None):
    if process is None:
        process = psutil.Process(os.getpid())

    bytes = float(process.memory_info().rss)

    if bytes < 1024.0:
        return f"{bytes} bytes"

    kb = bytes / 1024.0
    if kb < 1024.0:
        return f"{kb} KB"

    return f"{kb / 1024.} MB"

def create_dir_after_check(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

from os.path import isabs, getmtime
from os import getcwd, listdir, path

def get_latest_file(p):
    if not isabs(p):
        p = path.join(getcwd(), p)
    files = sorted([path.join(p, x) for x in listdir(p)], key=getmtime)
    return (files and files[-1]) or None