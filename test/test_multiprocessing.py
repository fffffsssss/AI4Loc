from multiprocessing import Pool
import os
import time
import random


def test(a1):
    t_start = time.time()
    print("%s 开始执行，进程号为%d" % (a1, os.getpid()))
    time.sleep(random.random() * 2)
    t_stop = time.time()
    print("%s 执行完毕，耗时%.2f" % (a1, (t_stop - t_start)))


if __name__ == "__main__":
    p1 = Pool(3)
    for i in range(0, 10):
        p1.apply_async(test, args=(i,))

    print("-----start-----")
    p1.close()
    p1.join()
    print("------end------")
