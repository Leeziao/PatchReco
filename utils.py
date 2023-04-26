import time
import json
from config import *

def statusNotifier(func):
    def wrapper(*args, **kwargs):
        print(f'\033[32m++ {func.__name__} Started\033[0m')
        beginTime = time.time()
        func(*args, **kwargs)
        endTime = time.time()
        print(f'\033[31m-- [{round(endTime-beginTime, 2)}s] {func.__name__} Finished\033[0m')
    return wrapper

hpJ = json.loads(hpFilePath.read_text())
