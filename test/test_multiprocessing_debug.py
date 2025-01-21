import multiprocessing
from multiprocessing import Process
import time
import os

#这个函数是要丢给子进程去处理的
def task(name):
    print("子进程%s is running"%name)
    print("子进程%s pid is %s"%(name,os.getpid()))
    print('子进程%s parent is %s'%(name,os.getppid()))
    time.sleep(5)
    print("子进程%s is end"%name)

#这里是程序入口，run的时候也是创建了一个进程来执行，这个是主进程
if __name__ =="__main__":
    #显示当前进程的名字
    print('当前进程名称是%s'%multiprocessing.current_process().name)
    print("当前进程的id is %s" % os.getpid())
    print("%s的父进程id是%s"%(multiprocessing.current_process().name,os.getppid()))
    print("我会创建一个子进程task")
    print('\n')

    p1 = Process(target=task,args=("task进程",))
    p2 = Process(target=task, args=("task进程",))
    #run() 方法并不启动一个新线程，就是在主线程中调用了一个普通函数而已。
    p1.start()
    p2.start()
    p1.join()    #主进程等待子进程结束后再继续执行
    p2.join()

    print("%s结束"%multiprocessing.current_process().name)
