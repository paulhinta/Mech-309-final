import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
from sympy.plotting import plot3d_parametric_line

r1 = 1.87527632324985
r2 = 4.69409122046058
r3 = 7.855

x = sp.Symbol('x')
s = sp.Symbol('s')
q1 = sp.Symbol('q1')
q2 = sp.Symbol('q2')
q3 = sp.Symbol('q3')

def phi(x, r):
    return (sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(cos(r*x)-cosh(r*x)))

def d2phi(x, r):
    return (r**2)*(-sin(r*x) + sinh(r*x) + ((cos(r)+cosh(r))/(sin(r)+sinh(r)))*(-cos(r*x)-cosh(r*x)))

def psi_h(x):
    return q1*phi(x, r1) + q2*phi(x,r2) + q3*phi(x,r3)

def d2psi_h(x):
    return q1*d2phi(x, r1) + q2*d2phi(x, r2) + q3*d2phi(x, r3)

#function to single integrate
def h(x, r, q):
    return (phi(x,r)*d2psi_h(x))

def trapezoid(x, r, q, n=20, a=0, b=1):
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
k3 = k(x,s,r3)

f1 = trapezoid(x, r1, q1, 100) + 100*(double_trap(k1, 0, 1, x, 1, 20))
f2 = trapezoid(x, r2, q2, 100) + 100*(double_trap(k2, 0, 1, x, 1, 20))
f3 = trapezoid(x, r3, q3, 100) + 100*(double_trap(k3, 0, 1, x, 1, 20))

print(f1)
print(f2)
print(f3)

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

def BFGS(epsilon, n, x, A):
    x0 = np.array(x)                #pass q1q2 here
    A0 = A                          #pass matA here
    norm = n
    e = epsilon

    while norm > e:
        print(x0)
        q_1 = x0[0]
        q_2 = x0[1]
        q_3 = x0[2]

        f_1 = f1.subs([(q1, q_1), (q2, q_2), (q3, q_3)])
        f_2 = f2.subs([(q1, q_1), (q2, q_2), (q3, q_3)])
        f_3 = f3.subs([(q1, q_1), (q2, q_2), (q3, q_3)])
        f1f2f3 = np.array([f_1,f_2,f_3])

        delx = np.array(gauss3(A0, -1*f1f2f3))
        
        y = np.array([float(delx[0]), float(delx[1]), float(delx[2])])

        f_1 = f1.subs([(q1, q_1+y[0]), (q2, q_2+y[1]), (q3, q_3+y[2])])
        f_2 = f2.subs([(q1, q_1+y[0]), (q2, q_2+y[1]), (q3, q_3+y[2])])
        f_3 = f3.subs([(q1, q_1+y[0]), (q2, q_2+y[1]), (q3, q_3+y[2])])

        delf = np.array([f_1, f_2, f_3]) - f1f2f3

        print(delf)

        x0 = x0 + y

        #Update matrix A0

        #denom1
        denom = np.linalg.multi_dot([y, A0, np.transpose(y)])
        # i11 = (y[0])**2
        # i12 = (y[0]*y[1])
        # i13 = (y[0]*y[2])
        # i21 = (y[1]*y[0])
        # i22 = (y[1])**2
        # i23 = (y[1]*y[2])
        # i31 = (y[2]*y[0])
        # i32 = (y[2]*y[1])
        # i33 = (y[2])**2

        # inter = np.array([[i11, i12, i13],[i21, i22, i23],[i31, i32, i33]])
        
        term1 = (np.linalg.multi_dot([A0, np.transpose([y]), [y], A0]))/denom

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

        A0 = A0 -term1 + inter

        norm = np.linalg.norm(y)
        print(norm)

    return x0

q1q2 = [2, -0.8, 1]                                 #x0 vector; initial guess for q1, q2, q3
matA = np.array([[1,0,0],[0,1,0],[0,0,1]])          #A0 matrix; initial guess for A_k matrix (identity matrix)

eps  = 0.001         #epsilon
norm = 1             #norm

sol1 = BFGS(eps, norm, q1q2, matA)
print(sol1)

# q1, q2, q3 = [ 1.7774846  -0.84769887 -0.01233278]

def psi_s(x):
    return sol1[0]*phi(x, r1) + sol1[1]*phi(x,r2) + sol1[2]*phi(x, r3)

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

# plt(dx_t)

xs = np.arange(0, 1.01, 0.01)      #101 data points
ws = []
wl = []

for i in xs:
    ws.append(float(dx_t.subs(x, i)))
    wl.append(float(dx_l.subs(x, i)))

ws = np.array(ws)
wl = np.array(wl)

mplot.plot(xs, ws)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Transverse deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Transverse deflection at various positions along the beam with q1, q2, q3")
mplot.show()

mplot.plot(xs, wl)
mplot.xlabel("Position along the beam, x [m]")
mplot.ylabel("Longitudinal deflection of the beam, w(x) [m]")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Longitudinal deflection at various positions along the beam with q1, q2, q3")
mplot.show()

plot3d_parametric_line(x, dx_t, dx_l, (x, 0, 1),  title="Transverse & Longitudonal deflection of the Beam", xlabel="Position along beam [m]", ylabel="dx_lverse dx_t [m]", zlabel="Longitudinal dx_t [m]")