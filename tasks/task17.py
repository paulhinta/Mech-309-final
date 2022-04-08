from turtle import color
from numpy import average, cos, cosh, sin, sinh, sqrt
import numpy as np
import matplotlib.pyplot as plt

import time
'''
TASK 17
'''
#All the roots that we found
r1 = 1.87527632324985
r2 = 4.69409122046058
r3 = 7.855

q11 = 4.277993721123706

q21 = 1.75918596
q22 = -0.83586439

q31 = 1.77748462
q32 = -0.84753308
q33 = -0.01671216

#phi functions for different (q,r) combinations
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

n=100

#integral function in r1; this is the same integral approx that we have been using
def int1(x0):
    b = 1
    a = x0

    dx = (b-a)/n

    if dx==0:
        return 0

    total_s = 100*cos(psi_1(x0)-psi_1(a))
    total_s += 100*cos(psi_1(x0)-psi_1(b))

    ss = np.arange(a+dx, b, dx)
    for s in ss:
        total_s += 2*100*cos(psi_1(x0)-psi_1(s))

    return total_s*dx/2
    
#integral function in r2
def int2(x0):
    b = 1
    a = x0

    dx = (b-a)/n

    total_s = 100*cos(psi_2(x0)-psi_2(a))
    total_s += 100*cos(psi_2(x0)-psi_2(b))

    if dx==0:
        return 0

    ss = np.arange(a+dx, b, dx)
    for s in ss:
        total_s += 2*100*cos(psi_2(x0)-psi_2(s))

    return total_s*dx/2

#integral function in r3
def int3(x0):
    b = 1
    a = x0

    dx = (b-a)/n

    if dx==0:
        return 0

    total_s = 100*cos(psi_3(x0)-psi_3(a))
    total_s += 100*cos(psi_3(x0)-psi_3(b))

    ss = np.arange(a+dx, b, dx)
    for s in ss:
        total_s += 2*100*cos(psi_3(x0)-psi_3(s))

    return total_s*dx/2

#take values of the LHS of the ODE to plot
ode1 = []
ode2 = []
ode3 = []

xs = np.arange(0,1.01,0.01)

for x in xs:
    ode1.append(abs(d2psi_1(x) + int1(x)))
    ode2.append(abs(d2psi_2(x) + int2(x)))
    ode3.append(abs(d2psi_3(x) + int3(x)))

#send ode list to array
ode1 = np.array(ode1)
ode2 = np.array(ode2)
ode3 = np.array(ode3)

res1 = sqrt(average(ode1**2))
max1 = max(ode1)
res2 = sqrt(average(ode2**2))
max2 = max(ode2)
res3 = sqrt(average(ode3**2))
max3 = max(ode3)

plt.plot(xs, ode1, label="1D case", color="k")
plt.plot(xs, ode2, label="2D case", color="r")
plt.plot(xs, ode3, label="3D case", color="b")
plt.xlabel("Position along the beam, x [m]")
plt.ylabel("Absolute value of LHS of ODE")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Convergence of psi(x) using different numbers of qi terms")
plt.legend()
plt.show()

#Guess ROC
ith_term = [1,2,3]
q_ith = [q11, abs(q22), abs(q33)]

plt.plot(ith_term, q_ith, '--*')
plt.xlabel("Number of q terms")
plt.ylabel("Absolute value of q_ith term")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Approximation of Convergence of psi(x) using i q terms")
plt.show()

#residual values (average & max)
print("Average residual of 1D case")
print(res1)
print("Max residucal of 1D case")
print(max1)
print("\n")
print("Average residual of 2D case")
print(res2)
print("Max residucal of 2D case")
print(max2)
print("\n")
print("Average residual of 3D case")
print(res3)
print("Max residucal of 3D case")
print(max3)
print("\n")