from concurrent.futures import ThreadPoolExecutor
from time import sleep

import cupy

# overlapping
# size = 1000000
# cycles = 1000
# streams_num = 8

size = 1000000
cycles = 1000
streams_num = 8

streams = [cupy.cuda.Stream() for _ in range(streams_num)]

with cupy.cuda.Stream():
    x = cupy.ones((size,))


def f(stream):
    global x
    with stream:
        for i in range(cycles):
            x += x


e = ThreadPoolExecutor(12)

list(e.map(f, streams))

sleep(1)
