import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot

#STEP 1: Find the second root q_2

x = sp.Symbol('x')
s = sp.Symbol('s')
q1 = sp.Symbol('q1')
q2 = sp.Symbol('q2')

r1 = 1.87527632324985
r2 = 4.69409122046058

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x):
    return q1*phi(x, r1) + q2*phi(x,r2)

def d2psi_h(x):
    return q1*d2phi(x, r1) + q2*d2phi(x, r2)

#function to single integrate
def h(x, r, q):
    return q*(phi(x,r)*d2psi_h(x))

def trapezoid(x, r, q, n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a, r, q) + h(b, r, q)

    for i in range(1, n):
        total += 2*h(a + dx*i, r, q)

    return const*total

def double_trap(func, x_low, x_high, s_low, s_high, n=10):
    #first pass in ds
    const_s = (s_high-s_low)/(2*n)

    ds = (s_high-s_low)/n

    total_s = func.subs(s, s_low) + func.subs(s, s_high)

    for i in range(1, n):
        total_s += 2*(func.subs(s, s_low + ds*i))

    total_s = const_s*total_s

    #second pass in dx
    const_x = (x_high-x_low)/(2*n)

    dx = (x_high-x_low)/n

    total_x = total_s.subs(x, x_low) + total_s.subs(x,x_high)

    for i in range(1, n):
        total_x += 2*(total_s.subs(x, x_low + dx*i))

    total_x = total_x*const_x

    return total_x

#function to double integrate
def k(x,s,r):
    return (phi(x,r))*(cos(psi_h(x) - psi_h(s)))