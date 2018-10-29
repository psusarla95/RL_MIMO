import numpy as np
from scipy.constants import *
import cmath
import math

def func(a):
    if a >  0:
        return 2, 3
    else:
        return "no"


if __name__ == '__main__':
    print(func(-2))
    print(func(6))
    val = func(6)
    a, b = val
    print(a)
    print(b)