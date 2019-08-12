from concurrent.futures import ThreadPoolExecutor
from time import sleep

import cupy

size = 512
cycles = 10
streams_num = 8

streams = [cupy.cuda.Stream() for _ in range(streams_num)]

with cupy.cuda.Stream():
    x = cupy.ones((size, size, 4))


def f(stream):
    global x
    with stream:
        for i in range(cycles):
            x += x


e = ThreadPoolExecutor(12)

list(map(f, streams))

sleep(1)
