import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
'''
TASK 3
'''
#Newton's method 
def newton(f,df,x_0,eps):
    xn = x_0

    while True:
        fxn = f.subs(x,xn)
        if abs(fxn) < eps:
            return xn
        dfxn = df.subs(x,xn)
        if dfxn == 0:
            print('Zero derivative -> no solution')
            return None
        xn = xn - fxn/dfxn

x = sp.Symbol("x")

#math functions of f & df/dx
def f(r):
    return cos(r)*cosh(r) + 1

def df(r):
    return -sin(r)*cosh(r) + cos(r)*sinh(r)

#input parameters for newton's method
eps = 0.001

r1 = newton(f(x), df(x), 1.5, eps)          #r1 = 1.87527632324985
r2 = newton(f(x), df(x), 4.25, eps)         #r2 = 4.69409122046058

print("r_1: {:.3f}".format(r1))
print("r_2: {:.3f}".format(r2))

def phi_1(x):
    return sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x))

def phi_2(x):
    return sin(r2*x) + sinh(r2*x) + ((cos(r2)+cosh(r2))/(sin(r2)+sinh(r2)))*(cos(r2*x)-cosh(r2*x))

x = np.arange(-2,3,0.02)    #250 data points
p1 = []                     #y values for phi_1
p2 = []                     #y values for phi_2

for point in x:
    p1.append(phi_1(float(point)))
    p2.append(phi_2(float(point)))

p1 = np.array(p1)
p2 = np.array(p2)

#plot phi_1
mplot.plot(x, p1, 'k')
mplot.title("Plot of phi_1")
mplot.show()

#plot phi_2
mplot.plot(x, p2, 'b')
mplot.title("Plot of phi_2")
#max point of phi_2 (for clarity)
xmax = round(x[np.argmax(p2)], 2)
ymax = round(p2.max(), 6)
def annot_max(x,y, ax=None):
    text= "Max point: x={:.2f}, y={:.6f}".format(xmax, ymax)
    if not ax:
        ax=mplot.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,p2)
mplot.ylim((-10000,1000))
mplot.show()