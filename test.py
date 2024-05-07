
import numpy as np


def fun(fun_a):
    fun_a[2] = 4
    fun_a.append(5)


if __name__ == '__main__':
    a = [1, 2, 3]
    fun(a)
    print(a)
