import torch
import torch.multiprocessing as mp
# import multiprocessing as mp
import time
import os
import numpy as np
import platform

import ailoc.common


def get_peak_memory_usage():
    """
    Get peak memory usage of current process in GB.
    """
    if platform.system() == 'Windows':
        import psutil
        peak_memory_usage = psutil.Process(os.getpid()).memory_info().peak_wset
    else:
        import resource
        peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Convert from KB to bytes

    return peak_memory_usage / (1024 * 1024 * 1024)  # Convert from bytes to GB


def producer(queue):
    print(f'enter the producer process: {os.getpid()}')
    wait_time = 0
    put_time = 0
    count = 100
    while count > 0:
        t0 = time.monotonic()
        if queue.full():
            # print('queue is full')
            wait_time += time.monotonic() - t0
            continue

        raw_data = np.ones([100, 512, 512], dtype=np.float32)  # read ndarray from file
        item = {
            'raw_data': torch.tensor(raw_data),  # convert raw data to tensor
            # 'raw_data': raw_data,  # directly use ndarray
            'sub_fov': [0, 1, 2, 3],
            'original_sub_fov': [4, 5, 6, 7],
            'frame': count
        }
        queue.put_nowait(item)  # put item into queue
        count -= 1

        put_time += time.monotonic() - t0
    queue.put(None)  # put sentinel value to indicate the end of the queue

    print(f'total producer wait time: {wait_time}')
    print(f'total put time: {put_time}')
    queue.join()

    # Record the peak memory usage of the producer process
    print(f'Process {os.getpid()} '
          f'peak RAM used: {get_peak_memory_usage()} GB')


def comsumer(queue):
    print(f'enter the comsumer process: {os.getpid()}')
    wait_time = 0
    get_time = 0
    gpu_time = 0
    anlz_time = 0
    while True:
        t0 = time.monotonic()
        if queue.empty():
            # print('queue is empty')
            wait_time += time.monotonic() - t0
            continue
        item = queue.get_nowait()
        queue.task_done()
        get_time += time.monotonic() - t0

        # Break if we get the sentinel value
        if item is None:
            break

        t1 = time.monotonic()
        raw_data = item['raw_data']
        sub_fov = item['sub_fov']
        original_sub_fov = item['original_sub_fov']
        frame = item['frame']
        data_for_network = ailoc.common.gpu(raw_data)  # move tensor to GPU
        gpu_time += time.monotonic() - t1
        # print(data_for_network.shape, sub_fov, original_sub_fov, frame)
        time.sleep(0.1)  # pseudo analysis time
        anlz_time += time.monotonic() - t1

    print(f'total comsumer wait time: {wait_time}')
    print(f'total get time: {get_time}')
    print(f'total gpu time: {gpu_time}')
    print(f'total analyze time: {anlz_time}')

    # Record the peak memory usage of the consumer process
    print(f'Process {os.getpid()} '
          f'peak RAM used: {get_peak_memory_usage()} GB')


if __name__ =="__main__":
    print(f'{mp.current_process().name}, id is {os.getpid()}')

    shared_queue = mp.JoinableQueue(maxsize=20)
    # shared_queue = mp.Queue(maxsize=20)

    producer_p = mp.Process(target=producer, args=(shared_queue, ))
    comsumer_p = mp.Process(target=comsumer, args=(shared_queue, ))

    t0 = time.monotonic()

    producer_p.start()
    # time.sleep(25)
    comsumer_p.start()

    producer_p.join()  # 主进程等待子进程结束后再继续执行
    comsumer_p.join()

    print(f"{mp.current_process().name} finished, total time: {time.monotonic() - t0}")

    # Record the peak memory usage of the main process
    print(f'Process {os.getpid()} '
          f'peak RAM used: {get_peak_memory_usage()} GB')
