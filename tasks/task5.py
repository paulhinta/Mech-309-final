import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
'''
TASK 5

'''
x = sp.Symbol('x')
q1 = sp.Symbol('q1')

r1 = 1.87527632324985

'''
First Integral
'''
def f(x):
    return (sin(r1*x)+sinh(r1*x) + (cos(r1*x)*cos(r1*x) - cosh(r1*x)*cosh(r1*x))/(sin(r1) + sinh(r1)))

def g(x):
    return (-sin(r1*x)+sinh(r1*x) + 2*(sin(r1*x)*sin(r1*x) - cos(r1*x)*cos(r1*x) + sinh(r1*x)*sinh(r1*x) + cosh(r1*x)*cosh(r1*x))/ (sin(r1) + sinh(r1)))

def h(x):
    return f(x)*g(x)

def trapezoid(x, n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a) + h(b)

    for i in range(1, n):
        total += 2*h(a + dx*i)

    return const*total

def integral(x, n=10):
    return q1*(r1**2)*trapezoid(x, n)

y = integral(x, 1000)

print(y)

#CHECK: Integral
#Formula for the online calculator:
#(sin(15x/8) + sinh(15x/8) + (cos^2(15x/8) -cosh^2(15x/8))/(sin(15/8) + sinh(15/8)))(sinh(15x/8) - sin(15x/8) + (2sin^2(15/8x) -2cos^2(15x/8) + 2sinh^2(15x/8) + 2cosh^2(15/8x))/(sin(15/8) + sinh(15/8)))
#online solution: 5.494730788809941
#Trapezoid solution: 5.49692283891431 (with n=100)
z = trapezoid(x, 100)
print(z)

'''
Second Integral
'''