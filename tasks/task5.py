import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh, sqrt
import numpy as np
import matplotlib.pyplot as mplot
'''
TASK 5

'''
x = sp.Symbol('x')
s = sp.Symbol('s')
q1 = sp.Symbol('q1')

r1 = 1.87527632324985
#r1 derived in previous task

'''
First Integral, steps 1-3
'''
def phi(x):
    return sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x))

def d2phi(x):
    return -sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(-cos(r1*x)-cosh(r1*x))

def h(x):
    return phi(x)*d2phi(x)

#STEP 3
#custom trapezoid rule solver
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

print(y)

#CHECK: Integral
#Formula for the online calculator:
#[sin(15x/8) + sinh(15x/8) + ((cos(15/8) + cosh(15/8))/(sin(15/8) + sinh(15/8)))(cos(15x/8) -cosh(15x/8))][-sin(15x/8) + sinh(15x/8) + ((cos(15/8) + cosh(15/8))/(sin(15/8) + sinh(15/8)))(-cos(15x/8) -cosh(15x/8)])*(225/64)
#online solution: −1.000055499072578 OR -3.515820113927032*q1
#Trapezoid solution: -0.999816351571781 OR -3.51601545922326*q1 (with n=100)
# z = trapezoid(x, 100)
# print(z)

'''
Second Integral, steps 4-5
'''
def k(x,s):
    return (phi(x))*(cos(q1*phi(x) - q1*phi(s)))

k = 100*k(x,s)

#STEP 5
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

    #now total_s is a representation of the first integral evaluated between the upper & lower bounds
    #total_s is a function of x & q1

    #second pass in dx
    const_x = (x_high-x_low)/(2*n)

    dx = (x_high-x_low)/n

    total_x = total_s.subs(x, x_low) + total_s.subs(x,x_high)

    for i in range(1, n):
        total_x += 2*(total_s.subs(x, x_low + dx*i))

    total_x = total_x*const_x

    return total_x

print("SECOND INTEGRAL \n")

z = double_trap(k, 0, 1, x, 1, 10)
print(z)

#sanity check
# test = (s**2)*x
# test_result = double_trap(test, 0, 1, x, 3, 10)
# print(test_result)
#RESULT: 4.44309695000000 (with n=10) 
#result from symbolab: 4.43333...
#the double trapezoid function works well

'''
f(q1), steps 6-7
'''
def f_q():
    return y + z

fq = f_q()

q = np.arange(-10,10,0.05)      #400 data points
p1 = []                         #f(q) values the points

#fixed point
fixed_p = []        #f(q) function

for point in q:
    p1.append(fq.subs(q1, float(point)))    #evaluate f(q_1) at that point

    fixed_p.append(z.subs(q1, float(point))/3.51601545922326)


p1 = np.array(p1)

fixed_p = np.array(fixed_p)

mplot.plot(q, p1, 'k')
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Plot of f(q_1)")
mplot.show()
'''
TASK 6
'''

#mplot.plot(q, p1, 'k', label='f(q_1)')
mplot.plot(q, fixed_p, label='g(q_1)')
mplot.plot(q, q, 'r', label='q_1')
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.axvline(x=2.75, ymin=0.35, ymax=0.7, color='g', linestyle=':', linewidth=2)
mplot.axvline(x=4.5, ymin=0.35, ymax=0.7, color='g', linestyle=':', linewidth=2)
mplot.axhline(y=2.75, xmin=0.35, xmax=0.7, color='g', linestyle=':', linewidth=2)
mplot.axhline(y=4.5, xmin=0.35, xmax=0.7, color='g', linestyle=':', linewidth=2)
mplot.xlim(1, 6)
mplot.ylim(1,6)
mplot.legend()
mplot.show()

'''
TASK 7
Numerical approximation of q_1 using fixed-point iteration

y & z from before are the g & (f-g) functions
'''
def secant(r0, r1, e):
    r = r1 - fq.subs(q1, r1)*((r1-r0)/(fq.subs(q1,r1) - fq.subs(q1,r0)))

    while abs(r-r1) > e:
        r0 = r1
        r1 = r
        r = r1 - fq.subs(q1, r1)*((r1-r0)/(fq.subs(q1,r1) - fq.subs(q1,r0)))

    return r

root = secant(2.75, 3, 0.001)

print(root)

#result: q1=3.27627371801516

#Convergence of integral function
ns = [1, 3, 5, 10, 15, 20]
solss = []
erss = []

for i in ns:
    sol = (trapezoid(x, i, 0, 1))*r1**2
    solss.append(float(sol))

    error = float(abs((sol - y.subs(q1, 1))/y.subs(q1, 1)))
    erss.append(error)

ns = np.array(ns)
solss = np.array(solss)
erss = np.array(erss)

mplot.plot(ns, solss, '--*')
mplot.title("Convergence of Single Integral Trapezoid Rule Approximation")
mplot.xlabel("Number of iterations, n")
mplot.ylabel("Integral Approximation")
mplot.show()

mplot.plot(ns, erss, '--*')
mplot.title("Convergence of Single Integral Trapezoid Rule Approximation")
mplot.xlabel("Number of iterations, n")
mplot.ylabel("Relative error")
mplot.show()