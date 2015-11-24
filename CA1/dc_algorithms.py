# -*- coding: utf-8 -*-
from collections import namedtuple
# import numpy as np
import doctest

# A = np.random.random((8,8));
# B = np.random.random((8,8));

# A = np.ones((8,8));
# B = np.ones((8,8));

A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
B = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]

SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]

max_sub_array_location = namedtuple("max_sub_array_location",
                                    "store_buy_day store_sell_day max_sum")


def find_maximum_subarray_brute(SPC, low, high):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Implement the brute force method from chapter 4
    time complexity = O(n^2)

    >>> find_maximum_subarray_brute(SPC, 0, len(SPC) - 1)
    (7, 10)
    """
    max = 0
    store_buy_day = 0
    store_sell_day = 0
    buy_day = 0

    while buy_day <= high:

        sum = SPC[buy_day]

        sell_day = buy_day + 1

        while sell_day <= high:

            if buy_day < sell_day:

                sum += SPC[sell_day]

                if sum >= max:
                    max = sum
                    store_buy_day = buy_day
                    store_sell_day = sell_day

            sell_day += 1
        buy_day += 1

    return store_buy_day, store_sell_day


def find_maximum_crossing_subarray(SPC, low, mid,  high):
    """
    Find the maximum subarray that crosses mid
    Return a tuple (i,j) where A[i:j] is the maximum subarray.

    >>> find_maximum_crossing_subarray(SPC, 0, len(SPC) - 1 / 2,  len(SPC) - 1)
    (7, 10)
    """
    tuple = maximum_crossing_subarray(SPC, low, mid,  high)

    return tuple.store_buy_day, tuple.store_sell_day


def maximum_crossing_subarray(SPC, low, mid,  high):

    low = 0
    high = len(SPC) - 1
    mid = low + high / 2

    left_sum = -9999
    right_sum = -9999

    max_left = 0
    max_right = 0
    sum = 0

    i = mid

    while i >= low:

        sum += SPC[i]

        if sum > left_sum:

            left_sum = sum
            max_left = i
        i -= 1

    sum = 0

    j = mid + 1

    while j <= high:

        sum += SPC[j]

        if sum > right_sum:

            right_sum = sum
            max_right = j
        j += 1

    return max_sub_array_location(store_buy_day=max_left,
                                  store_sell_day=max_right,
                                  max_sum=(left_sum + right_sum))


def find_maximum_subarray_recursive(SPC, low, high):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Recursive method from chapter 4

    >>> find_maximum_subarray_recursive(SPC, 1, len(SPC))
    (7, 10)
    """
    tuple = maximum_subarray_recursive(SPC, low, high)

    return tuple.store_buy_day, tuple.store_sell_day


def maximum_subarray_recursive(SPC, low, high):

    if high == low:

        return max_sub_array_location(store_buy_day=low,
                                      store_sell_day=high,
                                      max_sum=0)

    else:

        mid = (low+high) / 2

        left = maximum_subarray_recursive(SPC, low, mid)
        right = maximum_subarray_recursive(SPC, mid+1, high)
        cross = maximum_crossing_subarray(SPC, low, mid, high)

        if left.max_sum >= right.max_sum and left.max_sum >= cross.max_sum:
            return left

        elif right.max_sum >= left.max_sum and right.max_sum >= cross.max_sum:
            return right
        else:
            return cross


def find_maximum_subarray_iterative(SPC, low, high):
    """
    Return a tuple (i,j) where A[i:j] is the maximum subarray.
    Do problem 4.1-5 from the book.

    >>> find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)
    (7, 10)
    """
    max_sum = 0
    best_sum = 0
    current_sum = 0
    current_buy_day = -1
    best_buy_day = -1
    best_sell_day = -1

    i = 0

    while i <= high:

        current_sum = max_sum + SPC[i]

        if max_sum == 0:
            current_buy_day = i

        if current_sum > 0:
            max_sum = current_sum
        else:
            max_sum = 0

        if max_sum > best_sum:
            best_sum = max_sum
            best_sell_day = i
            best_buy_day = current_buy_day

        i += 1
    return best_buy_day, best_sell_day


def square_matrix_multiply(A, B):
    """
    Return the product AB of matrix multiplication.

    >>> square_matrix_multiply(A, B)
    [[10, 20, 30, 40], [26, 52, 78, 104], [42, 84, 126, 168], [58, 116, 174, 232]]
    """
    C = new_matrix(len(A), len(B[0]))

    if len(A[0]) != len(B):
        return "Can not multiply matrices"
    else:
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    C[i][j] += A[i][k]*B[k][j]
        return C


def square_matrix_multiply_strassens(A, B):
    """
    Return the product AB of matrix multiplication.
    Assume len(A) is a power of 2

     >>> square_matrix_multiply_strassens(A, B)
     [[10, 20, 30, 40], [26, 52, 78, 104], [42, 84, 126, 168], [58, 116, 174, 232]]
    """
    matrix_size = len(A)

    if matrix_size == 1:
        d = [[0]]
        d[0][0] = A[0][0] * B[0][0]
        return d
    else:

        a11, a12, a21, a22 = divide_matrix(A)
        b11, b12, b21, b22 = divide_matrix(B)

        p1 = square_matrix_multiply_strassens(add_matrix(a11, a22),
                                              add_matrix(b11, b22))
        p2 = square_matrix_multiply_strassens(add_matrix(a21, a22), b11)
        p3 = square_matrix_multiply_strassens(a11, subtract_matrix(b12, b22))
        p4 = square_matrix_multiply_strassens(a22, subtract_matrix(b21, b11))
        p5 = square_matrix_multiply_strassens(add_matrix(a11, a12), b22)
        p6 = square_matrix_multiply_strassens(subtract_matrix(a21, a11),
                                              add_matrix(b11, b12))
        p7 = square_matrix_multiply_strassens(subtract_matrix(a12, a22),
                                              add_matrix(b21, b22))

        c11 = add_matrix(subtract_matrix(add_matrix(p1, p4), p5), p7)
        c12 = add_matrix(p3, p5)
        c21 = add_matrix(p2, p4)
        c22 = add_matrix(subtract_matrix(add_matrix(p1, p3), p2), p6)

        C = new_matrix(len(c11)*2, len(c11)*2)

        for i in range(len(c11)):
            for j in range(len(c11)):
                C[i][j] = c11[i][j]
                C[i][j+len(c11)] = c12[i][j]
                C[i+len(c11)][j] = c21[i][j]
                C[i+len(c11)][j+len(c11)] = c22[i][j]

        return C


def new_matrix(p, q):
    return [[0 for row in range(p)] for col in range(q)]


def divide_matrix(M):

    new_size = len(M)/2

    a = new_matrix(new_size, new_size)
    b = new_matrix(new_size, new_size)
    c = new_matrix(new_size, new_size)
    d = new_matrix(new_size, new_size)

    for i in range(0, new_size):
            for j in range(0, new_size):
                a[i][j] = M[i][j]
                b[i][j] = M[i][j + new_size]
                c[i][j] = M[i + new_size][j]
                d[i][j] = M[i + new_size][j + new_size]

    return a, b, c, d


def add_matrix(a, b):
    if type(a) == int:
        d = a + b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] + b[i][j])
            d.append(c)
    return d


def subtract_matrix(a, b):
    if type(a) == int:
        d = a - b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] - b[i][j])
            d.append(c)
    return d


def test():

    print "Max sub array brute force :"
    find_maximum_subarray_brute_test()
    print "Max crossing sub array :"
    find_maximum_crossing_subarray_test()
    print "Max sub array recursive :"
    find_maximum_subarray_recursive_test()
    print "Max sub array iterative :"
    find_maximum_subarray_iterative_test()
    print "Matrix multiply iterative :"
    square_matrix_multiply_test()
    print "Matrix multiply using strassen's method :"
    square_matrix_multiply_strassens_test()

    doctest.testmod()


def find_maximum_subarray_brute_test():

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_brute(SPC, 0, len(SPC)-1)

    SPC = [13, -3, 25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_brute(SPC, 0, len(SPC)-1)

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, 22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_brute(SPC, 0, len(SPC)-1)


def find_maximum_crossing_subarray_test():

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_crossing_subarray(SPC, 0, len(SPC) - 1 / 2,  len(SPC) - 1) # flake8: noqa

    SPC = [13, -3, 25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_crossing_subarray(SPC, 0, len(SPC) - 1 / 2,  len(SPC) - 1) # flake8: noqa

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, 22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_crossing_subarray(SPC, 0, len(SPC) - 1 / 2,  len(SPC) - 1) # flake8: noqa


def find_maximum_subarray_recursive_test():

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_recursive(SPC, 0, len(SPC) - 1)

    SPC = [13, -3, 25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_recursive(SPC, 0, len(SPC) - 1)

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, 22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_recursive(SPC, 0, len(SPC) - 1)


def find_maximum_subarray_iterative_test():

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)

    SPC = [13, -3, 25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)

    SPC = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, 22, 15, -4, 7]
    print "INPUT :"
    print SPC
    print "OUTPUT :"
    print find_maximum_subarray_iterative(SPC, 0, len(SPC) - 1)


def square_matrix_multiply_test():

    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply(A, B)

    A = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    B = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply(A, B)

    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 4, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply(A, B)


def square_matrix_multiply_strassens_test():

    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply_strassens(A, B)

    A = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    B = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply_strassens(A, B)

    A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    B = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 4, 4, 4]]
    print "INPUT :"
    print A
    print B
    print "OUTPUT :"
    print square_matrix_multiply_strassens(A, B)

if __name__ == '__main__':

    test()