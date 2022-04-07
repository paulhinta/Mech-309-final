import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
from sympy.plotting import plot3d_parametric_line

#STEP 1: Find the second root q_2

x = sp.Symbol('x')
s = sp.Symbol('s')
q1 = sp.Symbol('q1')
q2 = sp.Symbol('q2')

r1 = 1.87527632324985
r2 = 4.69409122046058

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x):
    return q1*phi(x, r1) + q2*phi(x,r2)

def d2psi_h(x):
    return q1*d2phi(x, r1) + q2*d2phi(x, r2)

#function to single integrate
def h(x, r, q):
    return (phi(x,r)*d2psi_h(x))

def trapezoid(x, r, q, n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a, r, q) + h(b, r, q)

    for i in range(1, n):
        total += 2*h(a + dx*i, r, q)

    return const*total

def double_trap(func, x_low, x_high, s_low, s_high, n=10):
    #first pass in ds
    const_s = (s_high-s_low)/(2*n)

    ds = (s_high-s_low)/n

    total_s = func.subs(s, s_low) + func.subs(s, s_high)

    for i in range(1, n):
        total_s += 2*(func.subs(s, s_low + ds*i))

    total_s = const_s*total_s

    #second pass in dx
    const_x = (x_high-x_low)/(2*n)

    dx = (x_high-x_low)/n

    total_x = total_s.subs(x, x_low) + total_s.subs(x,x_high)

    for i in range(1, n):
        total_x += 2*(total_s.subs(x, x_low + dx*i))

    total_x = total_x*const_x

    return total_x

#function to double integrate
def k(x,s,r):
    return (phi(x,r))*(cos(psi_h(x) - psi_h(s)))

k1 = k(x,s,r1)
k2 = k(x,s,r2)

f1 = trapezoid(x, r1, q1, 100) + 100*(double_trap(k1, 0, 1, x, 1, 20))
f2 = trapezoid(x, r2, q2, 100) + 100*(double_trap(k2, 0, 1, x, 1, 20))

# print("f_1:")
# print(f1)
# print("\n")
# print("f_2:")
# print(f2)
'''
splot.plot3d(f1, (q1, -5, 5), (q2, -5, 5), title="f1(q1,q2) over interval of interest")
splot.plot3d(f2, (q1, -5, 5), (q2, -5, 5), title="f2(q1,q2) over interval of interest")
splot.plot3d(f1, f2, (q1, -5, 5), (q2, -5, 5), title="f1(q1,q2) and f2(q1,q2) over interval of interest")
'''

#relevant constants
#c1 = -3.51601545922326
#c2 = -0.00165774333088734
#k1 = -0.000349549292263339
#k2 = -22.0274691729771
#C = 2.06623440265869
#K = 0.431678350385201
#det(A) = 77.44876

'''
TASK 12
'''
q1q2 = [4.25, 0.3]                     #x0 vector; initial guess for q1, q2
matA = np.array([[1,0],[0,1]])         #A0 matrix; initial guess for A_k matrix (identity matrix)

eps  = 0.001         #epsilon
norm = 1             #norm

#function to sovle Ay = -f in 2D
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

def BFGS(epsilon, n, x, A):
    x0 = np.array(x)                #pass q1q2 here
    A0 = A                          #pass matA here
    norm = n
    e = epsilon

    while norm > e:
        print(x0)
        q_1 = x0[0]
        q_2 = x0[1]

        f_1 = f1.subs([(q1, q_1), (q2, q_2)])
        f_2 = f2.subs([(q1, q_1), (q2, q_2)])
        f1f2 = np.array([f_1,f_2])

        delx = np.array(gauss2(A0, -1*f1f2))
        
        y = np.array([float(delx[0]), float(delx[1])])

        f_1 = f1.subs([(q1, q_1+y[0]), (q2, q_2+y[1])])
        f_2 = f2.subs([(q1, q_1+y[0]), (q2, q_2+y[1])])

        delf = np.array([f_1, f_2]) - f1f2

        x0 = x0 + y

        #Update matrix A0

        #denom1
        denom = np.linalg.multi_dot([y, A0, np.transpose(y)])
        
        i11 = (y[0])**2
        i12 = (y[0]*y[1])
        i21 = (y[1]*y[0])
        i22 = (y[1])**2

        inter = np.array([[i11, i12],[i21, i22]])
        
        term1 = (np.linalg.multi_dot([A0, np.transpose([y]), [y], A0]))/denom

        denom = np.vdot(y, delf)

        i11 = (delf[0])**2
        i12 = (delf[0])*(delf[1])
        i21 = (delf[1])*(delf[0])
        i22 = (delf[1])**2

        inter = inter = (np.array([[i11, i12],[i21, i22]]))/denom

        A0 = A0 -term1 + inter

        norm = np.linalg.norm(y)
        print(norm)

    return x0

def broyden(epsilon, n, x, A):
    x0 = np.array(x)                #pass q1q2 here
    A0 = A                          #pass matA here
    norm = n
    e = epsilon

    while norm > e:
        #print(x0)
        q_1 = x0[0]
        q_2 = x0[1]

        f_1 = f1.subs([(q1, q_1), (q2, q_2)])
        f_2 = f2.subs([(q1, q_1), (q2, q_2)])
        f1f2 = np.array([f_1,f_2])

        delx = np.array(gauss2(A0, -1*f1f2))
        
        y = np.array([float(delx[0]), float(delx[1])])

        f_1 = f1.subs([(q1, q_1+y[0]), (q2, q_2+y[1])])
        f_2 = f2.subs([(q1, q_1+y[0]), (q2, q_2+y[1])])

        delf = np.array([f_1, f_2]) - f1f2

        x0 = x0 + y

        #Update matrix A0

        #term 1
        #print(A0)
        #print(y)
        ay = np.linalg.multi_dot([A0, y])
        #print(ay)
        t1 = delf - ay

        i11 = t1[0]*y[0]
        i12 = t1[0]*y[1]
        i21 = t1[1]*y[0]
        i22 = t1[1]*y[1]

        inter = np.array([[i11, i12],[i21, i22]])

        norm = np.linalg.norm(y)

        A0 = A0 + inter/(norm**2)

        #print(norm)

        #break

    return x0

#Try Broyden & BFGS
sol1 = BFGS(eps, norm, q1q2, matA)
#sol2 = broyden(eps, norm, q1q2, matA)
# print(sol1)
# print(sol2)
# [ 1.76556786 -0.83363552] bfgs
# [ 1.76555141 -0.8336386 ] broyden

'''
TASK 13
'''
def psi_s(x):
    return sol1[0]*phi(x, r1) + sol1[1]*phi(x,r2)

def weight_trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = sin(psi_s(a)) + sin(psi_s(b))

    for i in range(1, n):
        total += 2*sin(psi_s(a + dx*i))

    return const*total

def long_trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = cos(psi_s(a)) + cos(psi_s(b))

    for i in range(1, n):
        total += 2*cos(psi_s(a + dx*i))

    return const*total - x

dx_t = weight_trapezoid(100, 0, x)
dx_l = long_trapezoid(100, 0, x)

# plt(deflection)

xs = np.arange(0, 1.01, 0.01)      #101 data points
ws = []
wl = []

for i in xs:
    ws.append(float(dx_t.subs(x, i)))
    wl.append(float(dx_l.subs(x, i)))

ws = np.array(ws)

mplot.plot(xs, ws)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Transverse deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Transverse deflection at various positions along the beam with q1, q2")
mplot.show()

mplot.plot(xs, wl)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Longitudonal deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Longitudonal deflection at various positions along the beam with q1, q2")
mplot.show()

plot3d_parametric_line(x, dx_t, dx_l, (x, 0, 1),  title="Transverse & Longitudonal Deflection of the Beam", xlabel="Position along beam [m]", ylabel="Transverse Deflection [m]", zlabel="Longitudinal Deflection [m]")