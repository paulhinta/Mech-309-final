import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
import sympy.plotting as splot

#STEP 1: Find the second root q_2

x = sp.Symbol('x')
s = sp.Symbol('s')
q1 = sp.Symbol('q1')
q2 = sp.Symbol('q2')

r1 = 1.87527632324985
r2 = 4.69409122046058

f = 3.5*q1

#here, a = delx^T, np.transpose([a]) = delx, b = A_k
a = np.array([1, 3])
b = np.array([[1, 2], [2, 4]])

print(np.transpose(a))

#vector dot product
print(np.vdot(a, a))

#matrix dot product
print(np.dot(a, b))

#second term numerator
arrays = [b, np.transpose([a]), [a], b]
#print(arrays)
#matrix dot product, multiple array multiplication
print(np.linalg.multi_dot(arrays))

#second term denominator
arrays = [[a], b, np.transpose([a])]
print(np.linalg.multi_dot(arrays)[0][0])

#third term denominator

#GE solver in 2d
def gauss2(A, y):
    a11 = A[0][0]
    a12 = A[0][1]
    a21 = A[1][0]
    a22 = A[1][1]

    y1 = y[0]
    y2 = y[1]

    a12 = a12/a11
    y1 = y1/a11
    a11 = a11/a11       #set a11 = 1

    mult = a21/a11

    a21 = a21 - mult*a11    #set a21 = 0
    a22 = a22 - mult*a12
    y2 = y2 - mult*y1

    mult = a12/a22

    a12 = a12 - mult*a22    #set a12 = 0
    y1 = y1 - mult*y2
    y2 = y2/a22

    return [y1, y2]

mat = np.array([[9,5],[4,11]])
vec = np.array([4,1])

print(gauss2(mat, vec))

x0 = [3, 3]
delf = np.array([1, 2])