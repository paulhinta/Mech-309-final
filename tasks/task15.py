from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt

import time
'''
TASK 15
'''
r1 = 1.87527632324985
r2 = 4.69409122046058
r3 = 7.855

q1 = 1.77748462
q2 = -0.84753308
q3 = -0.01671216

#r1 and q1 derived in previous task
def phi(x, r):
    return sin(r*x) + sinh(r*x) + ((cos(r) + cosh(r))/(sin(r) + sinh(r)))*(cos(r*x)-cosh(r*x))

def psi(s, qi, qj, qk, ri, rj, rk):
    return qi*phi(s, ri) + qj*phi(s,rj) + qk*phi(s,rk)

def trans_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = sin(psi(a,q1, q2, q3, r1, r2, r3)) + sin(psi(b,q1, q2, q3, r1, r2, r3))

    for i in range(1, n):
        total += 2*sin(psi(a+i*dx, q1, q2, q3, r1, r2, r3))

    return const*total

def long_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = cos(psi(a, q1, q2, q3, r1, r2, r3)) + cos(psi(b, q1, q2, q3, r1, r2, r3))

    for i in range(1, n):
        total += 2*cos(psi(a+i*dx, q1, q2, q3, r1, r2, r3))

    return const*total

xs = np.arange(0, 1.01, 0.01)   #101 data points
trans_deflection = []
long_deflection = []

for x in xs:
    trans_deflection.append(trans_trapezoid(x))
    long_deflection.append(long_trapezoid(x) - x)

#plot trans deflection
plt.plot(xs, np.array(trans_deflection))
plt.xlabel("Position along the beam, x [m]")
plt.ylabel("Transverse deflection of the beam, w(x) [m]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Transverse deflection at various positions along the beam")
plt.show()

#plot long deflection
plt.plot(xs, np.array(long_deflection))
plt.xlabel("Position along the beam, x [m]")
plt.ylabel("Longitudinal deflection of the beam, u(x) [m]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Longitudinal deflection at various positions along the beam")
plt.show()

#plot overall deflection
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xs, long_deflection, trans_deflection, label="overall deflection of the beam")
ax.set_xlabel("Position along the beam, x [m]")
ax.set_zlabel("w(x) [m]")
ax.set_ylabel("u(x) [m]")
ax.legend()
plt.show()