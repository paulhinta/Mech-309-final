from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt

import time

from sympy import sec
'''
TASK 9
'''
r1 = 1.87527632324985

#r1 and q1 derived in previous task
#first, we must resolve for q1; each value of FN will dictate what q1 is
def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x, qi):
    return qi*phi(x, r1)

def d2psi_h(x, qi):
    return qi*d2phi(x, r1)

def trapezoid(r, qx, a=0, b=1, m=100):
    dx = (b-a)/m

    total = phi(a, r)*(d2psi_h(a, qx))
    total += phi(b, r)*(d2psi_h(b, qx))

    for i in range(1, m-1):
        total += 2*phi(a+i*dx, r)*(d2psi_h(a+i*dx, qx))
    
    return total*dx/2

def linear_slope(n=100, a=0, b=1):
    def h(x):
        return phi(x, r1)*d2phi(x, r1)
    
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a) + h(b)

    for i in range(1, n):
        total += 2*h(a + dx*i)

    return const*total

#double integral, same as previous questions
def double_int(r, qx, a=0, b=1, ups=1, m=25):
    dx = (b-a)/m
    xs = np.arange(a, b, dx)

    total_x = []
    for x in xs:
        lws = x
        ds = (ups - lws)/m

        total_s = phi(x, r)*cos(psi_h(x, qx) -psi_h(lws, qx))
        total_s += phi(x, r)*cos(psi_h(x, qx) -psi_h(ups, qx))

        for i in range(1, m):
            total_s += 2*phi(x, r)*cos(psi_h(x, qx) -psi_h(lws + i*ds, qx))
        
        total_x.append(total_s*ds/2)
    
    ret = 2*sum(total_x) - total_x[0] - total_x[-1]
    return ret*dx/2

def f(r, qx, force):
    return force*double_int(r, qx) + trapezoid(r, qx)

#find q for each force
def secant(force, ra, rb, e):
    r = rb - f(r1, rb, force)*((rb-ra))/(f(r1,rb,force) - f(r1,ra,force))

    while abs(r-rb) > e:
        ra = rb
        rb = r
        r = rb - f(r1, rb, force)*((rb-ra))/(f(r1,rb,force) - f(r1,ra,force))

    return r

n=100
fn = np.arange(0, n+1, 1)
qs = []

for fi in fn:
    qs.append(secant(fi, 0, 0.1, 0.001))

# print(fn)
# print(qs)

trans_deflection = []

#compute the transverse deflection at x=L for each force FN from 0 to 100
def psi(q1):
    return q1*(sin(r1*1) + sinh(r1*1) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*1)-cosh(r1*1)))

for i in range(len(fn)):
    trans_deflection.append(psi(qs[i]))

plt.plot(fn, np.array(trans_deflection))
plt.xlabel("Magnitude of the External Force FN")
plt.ylabel("Angle of rotation of the beam at x=L, w(L) [rad]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Angle of rotation of the beam at the end vs Magnitude of External Force")
plt.show()