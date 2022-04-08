from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt

import time
'''
TASK 8
'''
r1 = 1.87527632324985
q1 = 4.277993721123706

#r1 and q1 derived in previous task

def psi(s):
    return q1*(sin(r1*s) + sinh(r1*s) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*s)-cosh(r1*s)))

def trans_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = sin(psi(a)) + sin(psi(b))

    for i in range(1, n):
        total += 2*sin(psi(a+i*dx))

    return const*total

def long_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = cos(psi(a)) + cos(psi(b))

    for i in range(1, n):
        total += 2*cos(psi(a+i*dx))

    return const*total

xs = np.arange(0, 1.01, 0.01)   #101 data points
trans_deflection = []
long_deflection = []

for x in xs:
    trans_deflection.append(trans_trapezoid(x))
    long_deflection.append(long_trapezoid(x) - x)

plt.plot(xs, np.array(trans_deflection))
plt.xlabel("Position along the beam, x [m]")
plt.ylabel("Transverse deflection of the beam, w(x) [m]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Transverse deflection at various positions along the beam")
plt.show()

plt.plot(xs, np.array(long_deflection))
plt.xlabel("Position along the beam, x [m]")
plt.ylabel("Longitudinal deflection of the beam, u(x) [m]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Longitudinal deflection at various positions along the beam")
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(xs, long_deflection, trans_deflection, label="overall deflection of the beam")
ax.set_xlabel("Position along the beam, x [m]")
ax.set_zlabel("w(x) [m]")
ax.set_ylabel("u(x) [m]")
ax.legend()
plt.show()

