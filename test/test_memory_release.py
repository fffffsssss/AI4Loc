import numpy as np
import gc


def func1():
    print("func1")
    tmp_array_list = [np.random.rand(100, 2048, 2048),np.random.rand(50, 2048, 2048)]
    tmp_array = np.random.rand(100, 2048, 2048)
    func2(tmp_array_list, tmp_array)

    tmp_array = func3(tmp_array)

    del tmp_array_list, tmp_array
    gc.collect()
    print('func1 end')


def func2(array_list, array):
    print("func2")
    # a = array_list[0].mean()
    # b = array_list[1].mean()
    # c = array.mean()
    del array_list, array
    gc.collect()

    tmp_array_func2 = np.random.rand(100, 2048, 2048)
    # del tmp_array_func2
    # gc.collect()
    print('func2 end')


def func3(tmp_array):
    print("func3")
    tmp_array_1 = tmp_array+1
    print('func3 end')
    return tmp_array_1


if __name__ == '__main__':
    func1()
    print('main end')