from concurrent.futures import ThreadPoolExecutor
from time import sleep

import cupy

size = 1000000
cycles = 10
streams_num = 2

streams = [cupy.cuda.Stream(null=False, non_blocking=True) for _ in range(streams_num)]

# with cupy.cuda.Stream():
x = cupy.ones((size, 1))


def f(stream):
    global x
    with stream:
        for i in range(cycles):
            x += x


e = ThreadPoolExecutor(12)

list(map(f, streams))

list(map(lambda s: s.synchronize(), streams))
