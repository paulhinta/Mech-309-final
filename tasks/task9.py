import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh, sqrt
import numpy as np
import matplotlib.pyplot as mplot
'''
TASK 8
'''
r1 = 1.87527632324985

x = sp.Symbol('x')
s = sp.Symbol('s')
F = sp.Symbol('F')
q1 = sp.Symbol('q1')

def phi(x):
    return sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x))

def d2phi(x):
    return -sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(-cos(r1*x)-cosh(r1*x))

def h(x):
    return phi(x)*d2phi(x)

def trapezoid(x, n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a) + h(b)

    for i in range(1, n):
        total += 2*h(a + dx*i)

    return const*total

def integral(x, n=10):
    return q1*(r1**2)*trapezoid(x, n)

y = integral(x, 100)

def k(x,s):
    return (phi(x))*(cos(q1*phi(x) - q1*phi(s)))

k = F*k(s,x)

#custom trapezoid rule solver in 2d
#variables of integration: x, s
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

z = double_trap(k, 0, 1, x, 1, 10)

#return the sum of y + z, substitute the specific value for F_N
def psi(val):
    ret = y + z
    return ret.subs(F, val)

Fs = np.arange(0, 101, 1)           #100 input values of F_N
psi_s = []                          #psi output values 
qs = []                             #value of q_1 at different F_N

def secant(f, r0, r1, e):
    r = r1 - f.subs(q1, r1)*((r1-r0)/(f.subs(q1,r1) - f.subs(q1,r0)))

    while abs(r-r1) > e:
        r0 = r1
        r1 = r
        r = r1 - f.subs(q1, r1)*((r1-r0)/(f.subs(q1,r1) - f.subs(q1,r0)))

    return r

for i in Fs:
    fq = psi(i)
    q = secant(fq, 0, 0.5, 0.001)
    qs.append(q)

for q in qs:
    func = q*phi(x)
    psi_s.append(float(func.subs(x, 1)))        #sub x = L = 1

print("q_1 @ F_N = 100: " + str(qs[100]))

psi_s = np.array(psi_s)
qs = np.array(qs)

mplot.plot(Fs, psi_s)
mplot.xlabel("Transverse Force, F_N")
mplot.ylabel("Angle of Rotation @ x=L")
mplot.title("Angle of Rotation at x=L vs the Transverse Force of a 1-D Beam")
mplot.show()
