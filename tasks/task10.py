import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot

#STEP 1: Find the second root q_2

x = sp.Symbol('x')
q = sp.Symbol('q')
s = sp.Symbol('s')

r1 = 1.87527632324985
r2 = 4.69409122046058

q1 = 4.26275024501704
q2 = 0

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def h(x, r):
    return (phi(x,r)*d2phi(x,r))

def trapezoid(x, r, n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a, r) + h(b, r)

    for i in range(1, n):
        total += 2*h(a + dx*i, r)

    return const*total

def integral(x, r, q, n=10):
    return q*(r**2)*trapezoid(x, r, n)

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

def k(x,s,r,q):
    return (phi(x,r))*(cos(q*phi(x,r) - q*phi(s,r)))

#set up equation for q2 using root 2
y2 = integral(x, r2, q, 100)
k = 100*k(x,s,r2,q)
z2 = double_trap(k, 0, 1, x, 1, 10)
f_q2 = y2 + z2

qs = np.arange(-5,5,0.05)        #200 data points, only take +ve root
p2 = []                         #f(q) values the points

for point in qs:
    p2.append(f_q2.subs(q, float(point)))    #evaluate f(q_2) at that point

p2 = np.array(p2)

def secant(r0, r1, fq, q, e=0.001):
    r = r1 - fq.subs(q, r1)*((r1-r0)/(fq.subs(q,r1) - fq.subs(q,r0)))

    while abs(r-r1) > e:
        r0 = r1
        r1 = r
        r = r1 - fq.subs(q, r1)*((r1-r0)/(fq.subs(q,r1) - fq.subs(q,r0)))

    return r

mplot.plot(qs, p2)
mplot.show()

q2 = secant(0.5, 1, f_q2, q)        #0.268362029434715

'''
2D EXPRESSIONS
'''

def psi_h(x):
    return q1*phi(x, r1) + q2*phi(x,r2)

def d2psi_h(x):
    return q1*d2phi(x, r1) + q2*d2phi(x, r2)

print(psi_h(x))
print("\n")
print(d2psi_h(x))