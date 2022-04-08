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
r3 = 7.855

#r1 and r2 derived in previous task
'''
TASK 10
'''
#pass r1 or r2 along with number that you want to integrate

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x, qi, qj, qk):
    return qi*phi(x, r1) + qj*phi(x,r2) + qk*phi(x,r3)

def d2psi_h(x, qi, qj, qk):
    return qi*d2phi(x, r1) + qj*d2phi(x, r2) + qk*d2phi(x, r3)

def trapezoid(r, qx, qy, qz, a=0, b=1, m=100):
    dx = (b-a)/m

    total = phi(a, r)*(d2psi_h(a, qx, qy, qz))
    total += phi(b, r)*(d2psi_h(b, qx, qy, qz))

    for i in range(1, m-1):
        total += 2*phi(a+i*dx, r)*(d2psi_h(a+i*dx, qx, qy, qz))
    
    return total*dx/2

def double_int(r, qx, qy, qz, a=0, b=1, ups=1, m=25):
    dx = (b-a)/m
    xs = np.arange(a, b, dx)

    total_x = []
    for x in xs:
        lws = x
        ds = (ups - lws)/m

        total_s = phi(x, r)*cos(psi_h(x, qx, qy, qz) -psi_h(lws, qx, qy, qz))
        total_s += phi(x, r)*cos(psi_h(x, qx, qy, qz) -psi_h(ups, qx, qy, qz))

        for i in range(1, m):
            total_s += 2*phi(x, r)*cos(psi_h(x, qx, qy, qz) -psi_h(lws + i*ds, qx, qy, qz))
        
        total_x.append(total_s*ds/2)
    
    ret = 2*sum(total_x) - total_x[0] - total_x[-1]
    return ret*dx/2

def f(r, qx, qy, qz):
    return 100*double_int(r, qx, qy, qz) + trapezoid(r, qx, qy, qz)

# qi = 1.7665
# qj = -0.8336
# qk = -0.0123
# f1 = f(r1, qi, qj, qk)
# f2 = f(r2, qi, qj, qk)
# f3 = f(r3, qi, qj, qk)

# print(f1)
# print(f2)
# print(f3)


'''
TASK 12
'''
#custom G-E solver in 3d
def gauss3(A, y):
    a11 = A[0][0]
    a12 = A[0][1]
    a13 = A[0][2]
    a21 = A[1][0]
    a22 = A[1][1]
    a23 = A[1][2]
    a31 = A[2][0]
    a32 = A[2][1]
    a33 = A[2][2]

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    a12 = a12/a11
    a13 = a13/a11
    y1 = y1/a11
    a11 = a11/a11       #set a11 = 1

    mult = a21/a11

    a21 = a21 - mult*a11    #set a21 = 0
    a22 = a22 - mult*a12
    a23 = a23 - mult*a13
    y2 = y2 - mult*y1

    mult = a31/a11          #set a31 = 0
    a31 = a31 - mult*a11
    a32 = a32 - mult*a12
    a33 = a33 - mult*a13
    y3 = y3 - mult*y1

    mult = a32/a22
    a32 = a32 - mult*a22
    a33 = a33 - mult*a23
    y3 = y3 - mult*y2

    #backward substitution
    y3 = y3/a33
    y2 = (y2 - a23*y3)/a22
    y1 = (y1 - a12*y2 - a13*y3)/a11

    return [y1, y2, y3]

def bfgs(e, n, x, A):
    norm = n
    x0 = np.array(x)
    a0 = A

    #this code mimics the pseudocode in chapter 4 for broyden's method
    while norm > e:
        f1 = f(r1, x0[0], x0[1], x0[2])
        f2 = f(r2, x0[0], x0[1], x0[2])
        f3 = f(r3, x0[0], x0[1], x0[2])

        vf = np.array([f1, f2, f3])

        y = gauss3(a0, -1*vf)

        fa = f(r1, y[0] + x0[0], y[1] + x0[1], y[2] + x0[2])
        fb = f(r2, y[0] + x0[0], y[1] + x0[1], y[2] + x0[2])
        fc = f(r3, y[0] + x0[0], y[1] + x0[1], y[2] + x0[2])

        delf = np.array([fa, fb, fc]) - vf

        x0 = x0 + y

        denom = np.linalg.multi_dot([y, a0, np.transpose(y)])
        i11 = (y[0])**2
        i12 = (y[0]*y[1])
        i13 = (y[0]*y[2])
        i21 = (y[1]*y[0])
        i22 = (y[1])**2
        i23 = (y[1]*y[2])
        i31 = (y[2]*y[0])
        i32 = (y[2]*y[1])
        i33 = (y[2])**2

        #a0 replacement (iteration) starts here
        inter = np.array([[i11, i12, i13],[i21, i22, i23], [i31, i32, i33]])

        term1 = (np.linalg.multi_dot([a0, np.transpose([y]), [y], a0]))/denom

        denom = np.vdot(y, delf)

        i11 = (delf[0])**2
        i12 = (delf[0])*(delf[1])
        i13 = (delf[0])*(delf[2])
        i21 = (delf[1])*(delf[0])
        i22 = (delf[1])**2
        i23 = (delf[1])*(delf[2])
        i31 = (delf[2])*(delf[0])
        i32 = (delf[2])*(delf[1])
        i33 = (delf[2])**2

        inter = inter = (np.array([[i11, i12, i13],[i21, i22, i23],[i31, i32, i33]]))/denom
        
        #a0 matrix is replaced
        a0 = a0 -term1 + inter

        norm = np.linalg.norm(y)

    return x0

#call the bgfs method
epsilon, n = 0.001, 1
x0 = [1.75, -0.8, 0]
a0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(bfgs(epsilon, n, x0, a0))

#q1, q2, q3 = [ 1.77748462 -0.84753308 -0.01671216]