import subprocess


def memory_used():
    return subprocess.call("nvidia-smi --query-gpu=memory.used --format=csv".split(" "))
