from pandas import array
import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh, sqrt
import numpy as np
import matplotlib.pyplot as mplot
from sympy.plotting import plot3d_parametric_line
'''
TASK 8
'''
r1 = 1.87527632324985
q1 = 4.29105998838015       #3.27 
#r_1 & q_1 value derived in task 7
def psi(x):
    return q1*(sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x)))

def trans_trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = sin(psi(a)) + sin(psi(b))

    for i in range(1, n):
        total += 2*sin(psi(a + dx*i))

    return const*total

def long_trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = sin(psi(a)) + sin(psi(b))

    for i in range(1, n):
        total += 2*sin(psi(a + dx*i))

    return const*total -x

x = sp.Symbol("x")
s = sp.Symbol("s")
F = sp.Symbol("F")

trans = trans_trapezoid(100, 0, x)
long = long_trapezoid(100, 0, x)

xs = np.arange(0, 1, 0.01)      #100 data points
ws = []
wl = []

for i in xs:
    ws.append(float(trans.subs(x, i)))
    wl.append(float(long.subs(x,i)))

ws = np.array(ws)
wl = np.array(wl)

mplot.plot(xs, ws)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Transverse deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Transverse deflection at various positions along the beam")
mplot.show()

mplot.plot(xs, wl)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Longitudinal deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Longitudinal deflection at various positions along the beam")
mplot.show()

plot3d_parametric_line(x, trans, long, (x, 0, 1),  title="Transverse & Longitudonal Deflection of the Beam", xlabel="Position along beam [m]", ylabel="Transverse Deflection [m]", zlabel="Longitudinal Deflection [m]")