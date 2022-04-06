import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
from sympy.plotting import plot as plt
from sympy import cos, cosh, sin, sinh, simplify
import numpy as np
import matplotlib.pyplot as mplot

x = sp.Symbol('x')
s = sp.Symbol('s')

r1 = 1.87527632324985
r2 = 4.69409122046058
r3 = 7.855

#Superimpose plots
q11 = 4.29105998838015

q21 = 1.76556786
q22 = -0.83363552

q31 = 1.7774846
q32 = -0.84769887
q33 = -0.01233278

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_1(x):
    return q11*phi(x, r1)

def d2psi_1(x):
    return q11*d2phi(x, r1)

def psi_2(x):
    return q21*phi(x, r1) + q22*phi(x,r2)

def d2psi_2(x):
    return q21*d2phi(x, r1) + q22*d2phi(x, r2)

def psi_3(x):
    return q31*phi(x, r1) + q32*phi(x,r2) + q33*phi(x,r3)

def d2psi_3(x):
    return q31*d2phi(x, r1) + q32*d2phi(x, r2) + q33*d2phi(x, r3)

#governing ODE: (1)*d2phi(x) - int_{x,1}(100*cos(psi(x)-psi(s)))ds = 0
#evaluate LHS of the ODE, set RHS to zero

h_1 = 100*cos(psi_1(x) - psi_1(s))

h_2 = 100*cos(psi_2(x) - psi_2(s))

h_3 = 100*cos(psi_3(x) - psi_3(s))

def trapezoid_1(var, h, n=100, a=x, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h.subs(var, a) + h.subs(var, b)

    for i in range(1, n):
        total += 2*h.subs(var, a + dx*i)

    return const*total + d2psi_1(x)

def trapezoid_2(var, h, n=100, a=x, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h.subs(var, a) + h.subs(var, b)

    for i in range(1, n):
        total += 2*h.subs(var, a + dx*i)

    return const*total + d2psi_2(x)

def trapezoid_3(var, h, n=100, a=x, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h.subs(var, a) + h.subs(var, b)

    for i in range(1, n):
        total += 2*h.subs(var, a + dx*i)

    return const*total + d2psi_3(x)

ode1 = trapezoid_1(s, h_1)
ode2 = trapezoid_2(s, h_1)
ode3 = trapezoid_3(s, h_1)

xs = np.arange(0,1.01,0.01)
o1 = []
o2 = []
o3 = []

for i in xs:
    o1.append(abs(ode1.subs(x, i)))
    o2.append(abs(ode2.subs(x, i)))
    o3.append(abs(ode3.subs(x, i)))

o1 = np.array(o1)
o2 = np.array(o2)
o3 = np.array(o3)

#plt(ode1, ode2, ode3, (x,0,1), legend=True)
mplot.plot(xs, o1, color='k', label='1D Approx')
mplot.plot(xs, o2, color='b', label='2D Approx')
mplot.plot(xs, o3, color='r', label='3D Approx')
mplot.xlabel("Position along beam, x")
mplot.ylabel("Value of LHS of the ODE")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("LHS of the ODE vs x obtained by using different number of qi terms")
mplot.legend()
mplot.show()

#Guess ROC
ith_term = [1,2,3]
q_ith = [q11, abs(q22), abs(q33)]

mplot.plot(ith_term, q_ith, '--*')
mplot.xlabel("Number of q terms")
mplot.ylabel("Absolute value of q_ith term")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Approximation of Convergence of psi(x) using i q terms")
mplot.show()