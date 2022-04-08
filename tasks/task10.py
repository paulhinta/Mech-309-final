from cProfile import label
from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
'''
TASK 5

'''
r1 = 1.87527632324985
r2 = 4.69409122046058

#r1 and r2 derived in previous task
'''
TASK 10
'''
#pass r1 or r2 along with number that you want to integrate

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x, qi, qj):
    return qi*phi(x, r1) + qj*phi(x,r2)

def d2psi_h(x, qi, qj):
    return qi*d2phi(x, r1) + qj*d2phi(x, r2)

def trapezoid(r, qx, qy, a=0, b=1, m=100):
    dx = (b-a)/m

    total = phi(a, r)*(d2psi_h(a, qx, qy))
    total += phi(b, r)*(d2psi_h(b, qx, qy))

    for i in range(1, m-1):
        total += 2*phi(a+i*dx, r)*(d2psi_h(a+i*dx, qx, qy))
    
    return total*dx/2

def double_int(r, qx, qy, a=0, b=1, ups=1, m=25):
    dx = (b-a)/m
    xs = np.arange(a, b, dx)

    total_x = []
    for x in xs:
        lws = x
        ds = (ups - lws)/m

        total_s = phi(x, r)*cos(psi_h(x, qx, qy) -psi_h(lws, qx, qy))
        total_s += phi(x, r)*cos(psi_h(x, qx, qy) -psi_h(ups, qx, qy))

        for i in range(1, m):
            total_s += 2*phi(x, r)*cos(psi_h(x, qx, qy) -psi_h(lws + i*ds, qx, qy))
        
        total_x.append(total_s*ds/2)
    
    ret = 2*sum(total_x) - total_x[0] - total_x[-1]
    return ret*dx/2

def f(r, qx, qy):
    return 100*double_int(r, qx, qy) + trapezoid(r, qx, qy)

# qi = 1.7665
# qj = -0.8336
# f1 = f(r1, qi, qj)
# f2 = f(r2, qi, qj)

# print(f1)
# print(f2)

n=100
lim = 5
q1 = np.arange(-lim, lim, 2*lim/n)
q2 = np.arange(-lim, lim, 2*lim/n)
f1 = []                                 #matrix to hold the value of f1(q1,q2) for all discrete points q1, q2
f2 = []

qx = []

#evaluate all the points
for qj in q2:
    print(qj)
    qx.append(qj)
    qy1 = []
    qy2 = []
    for qi in q1:
        qy1.append(f(r1, qi, qj))
        qy2.append(f(r2, qi, qj))

    f1.append(qy1)
    f2.append(qy2)

f1_p = np.array(f1)
f2_p = np.array(f2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(q1, q2)
Z = f1_p.reshape(X.shape)
ax.plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('f1(q1,q2)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(q1, q2)
Z = f2_p.reshape(X.shape)
ax.plot_surface(X, Y, Z,cmap='coolwarm', edgecolor='none', label="f2")
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('f2(q1,q2)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(q1, q2)
Z = f2_p.reshape(X.shape)
W = f1_p.reshape(X.shape)
ax.plot_surface(X, Y, Z,cmap='coolwarm', edgecolor='none', label="f2")
ax.plot_surface(X, Y, W,cmap='viridis', edgecolor='none', label="f1")
ax.set_xlabel('q1')
ax.set_ylabel('q2')
plt.show()

'''
TASK 12
'''
#custom G-E solver in 2d
def gauss2(A, y):
    a11 = A[0][0]
    a12 = A[0][1]
    a21 = A[1][0]
    a22 = A[1][1]

    y1 = y[0]
    y2 = y[1]

    a12 = a12/a11
    y1 = y1/a11
    a11 = a11/a11       #set a11 = 1

    mult = a21/a11

    a21 = a21 - mult*a11    #set a21 = 0
    a22 = a22 - mult*a12
    y2 = y2 - mult*y1

    mult = a12/a22

    a12 = a12 - mult*a22    #set a12 = 0
    y1 = y1 - mult*y2
    y2 = y2/a22

    return [y1, y2]

#qx is the line of all possible q's
def bfgs(e, n, x, A):
    norm = n
    x0 = np.array(x)
    a0 = A

    while norm > e:
        f1 = f(r1, x0[0], x0[1])
        f2 = f(r2, x0[0], x0[1])

        vf = np.array([f1, f2])

        y = gauss2(a0, -1*vf)

        fa = f(r1, y[0] + x0[0], y[1] + x0[1])
        fb = f(r2, y[0] + x0[0], y[1] + x0[1])

        delf = np.array([fa, fb]) - vf

        x0 = x0 + y

        denom = np.linalg.multi_dot([y, a0, np.transpose(y)])
        i11 = (y[0])**2
        i12 = (y[0]*y[1])
        i21 = (y[1]*y[0])
        i22 = (y[1])**2

        inter = np.array([[i11, i12],[i21, i22]])

        term1 = (np.linalg.multi_dot([a0, np.transpose([y]), [y], a0]))/denom

        denom = np.vdot(y, delf)

        i11 = (delf[0])**2
        i12 = (delf[0])*(delf[1])
        i21 = (delf[1])*(delf[0])
        i22 = (delf[1])**2

        inter = inter = (np.array([[i11, i12],[i21, i22]]))/denom
        
        a0 = a0 -term1 + inter

        norm = np.linalg.norm(y)

    return x0

epsilon, n = 0.001, 1
x0 = [4.25, 0.3]
a0 = np.array([[1,0],[0,1]])
print(bfgs(epsilon, n, x0, a0))

#q1,q2 = [ 1.75918596 -0.83586439]