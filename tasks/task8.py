import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh, sqrt
import numpy as np
import matplotlib.pyplot as mplot
'''
TASK 8
'''
r1 = 1.87527632324985
q1 = 4.26275024501704       #3.27 
#r_1 & q_1 value derived in task 7
def psi(x):
    return q1*(sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x)))

def trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = sin(psi(a)) + sin(psi(b))

    for i in range(1, n):
        total += 2*sin(psi(a + dx*i))

    return const*total

x = sp.Symbol("x")
s = sp.Symbol("s")
F = sp.Symbol("F")

deflection = trapezoid(100, 0, x)

xs = np.arange(0, 1, 0.01)      #100 data points
ws = []

for i in xs:
    ws.append(float(deflection.subs(x, i)))

ws = np.array(ws)

mplot.plot(xs, ws)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Transverse deflection of the beam, w(x) [m]")
mplot.title("Transverse deflection at various positions along the beam")
mplot.show()