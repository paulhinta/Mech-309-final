from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt
import time
'''
TASK 5

'''
r1 = 1.87527632324985
#r1 derived in previous task
'''
First Integral, steps 1-3
'''
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

def f(r, qx):
    return 100*double_int(r, qx) + trapezoid(r, qx)

def g(r, qx):
    return -(100*double_int(r, qx) + trapezoid(r, qx))/(linear_slope())

# qi = 4.27775
# f1 = f(r1, qi)

n=100
lim = 5
q1 = np.arange(-lim, lim, 2*lim/n)
f1 = []                                 #matrix to hold the value of f1(q1) for all discrete points q1, q2
g1 = []
task7 = []

#evaluate all the points
for qi in q1:
    f1.append(f(r1, qi))
    if qi>=0:
        g1.append(g(r1, qi))

g1.append(g(r1, q1[-1]+2*lim/n))

f1_p = np.array(f1)

plt.plot(q1, f1_p)
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Plot of f(q_1)")
plt.show()

'''
TASK 6
'''
#fixed-point shit
qs = np.arange(0,5,10/n)
plt.plot(qs, g1)
plt.plot(qs, qs, 'r')
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=2.5, ymin=0.3, ymax=0.42, color='g', linestyle=':', linewidth=2)
plt.axvline(x=4, ymin=0.3, ymax=0.42, color='g', linestyle=':', linewidth=2)
plt.axhline(y=2.5, xmin=0.51, xmax=0.79, color='g', linestyle=':', linewidth=2)
plt.axhline(y=4, xmin=0.51, xmax=0.79, color='g', linestyle=':', linewidth=2)
plt.title("Fixed-point quadrature scheme")
plt.show()

'''

'''