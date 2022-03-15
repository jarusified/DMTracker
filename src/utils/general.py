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

def get_sorted_files(p):
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    files = sorted([os.path.join(p, x) for x in os.listdir(p)], key=os.path.getmtime)
    return files    

def get_latest_file(p):
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    files = sorted([os.path.join(p, x) for x in os.listdir(p)], key=os.path.getmtime)
    return (files and files[-1]) or None