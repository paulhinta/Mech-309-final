import sympy as sp      #math library to represent functions, we will write our own problem solving expressions
#from sympy.plotting import plot
from sympy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as mplot
from scipy import polyval, polyfit

ith_term = [1,2,3]
q_ith = [4.29105998838015, abs(-0.8336), abs(-0.01233278)]

x = np.linspace(0, 1, 10)

mplot.plot(ith_term, q_ith, '--*')
mplot.xlabel("Number of q terms")
mplot.ylabel("Absolute value of q_ith term")
mplot.grid(color='k', linestyle='--', linewidth=0.5)
mplot.title("Approximation of Convergence of psi(x) using i q terms")
mplot.show()