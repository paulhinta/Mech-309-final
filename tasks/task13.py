from cProfile import label
from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

r1 = 1.87527632324985
r2 = 4.69409122046058

q1=1.75918596
q2=-0.83586439

'''
TASK 13
'''
def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def psi_h(s, qi, qj, ri, rj):
    return qi*phi(s, ri) + qj*phi(s,rj)

def trans_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = sin(psi_h(a, q1, q2, r1, r2)) + sin(psi_h(b, q1, q2, r1, r2))

    for i in range(1, n):
        total += 2*sin(psi_h(a+i*dx, q1, q2, r1, r2))

    return const*total

def long_trapezoid(b=1, n=100):
    a=0
    const = (b-a)/n
    dx = const/2

    total = cos(psi_h(a,q1,q2,r1,r2)) + cos(psi_h(b,q1,q2,r1,r2))

    for i in range(1, n):
        total += 2*cos(psi_h(a+i*dx,q1,q2,r1,r2))

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