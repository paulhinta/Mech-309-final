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
def phi(x):
    return sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*x)-cosh(r1*x))

def d2phi(x):
    return (r1**2)*(-sin(r1*x) + sinh(r1*x) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(-cos(r1*x)-cosh(r1*x)))

def h(x):
    return phi(x)*d2phi(x)

def trapezoid(n=10, a=0, b=1):
    const = (b-a)/(2*n)

    dx = (b-a)/(n)

    total = h(a) + h(b)

    for i in range(1, n):
        total += 2*h(a + dx*i)

    return const*total

# print("Numpy -- First integral (linear component)")
# start = time.time()
y = trapezoid(100)

n=100

qs = np.arange(-5, 5, (5 - (-5))/n)          #100 data points for now
xs = np.arange(0, 1, (1-0)/n)                #100 data points

first_integral = []

# print(qs)
# print(xs)

def integrand(x0, q0, s0):
    return phi(x0)*cos(q0*(phi(x0) - phi(s0)))

for q in qs:
    out = []
    for x in xs:
        dx = (1-x)/n

        total = integrand(x, q, x) + integrand(x, q, 1)

        s_values = np.arange(x+dx, 1, dx)

        for s in s_values:
            total += 2*integrand(x, q, s)

        total = total*dx/2

        out.append(total)
    
    first_integral.append(out)

# for q_list in first_integral:
#     print(q_list)

second_integral = []
task6_LHS = []
task7 = []

j=0
dq = 10/n
for q in range(n):
    q0 = -5 + j*dq

    lst = first_integral[q]
    dx = 1/n
    total = lst[0] + lst[n-1]

    for i in range(1, n-2):
        total += lst[i]*2

    out = 100*(total*dx/2)

    second_integral.append(out + y*q0)
    if q0 >= 0:
        task6_LHS.append(-out/y)
        task7.append(out + y*q0)
    j+=1

plt.plot(qs, second_integral)
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Plot of f(q_1)")
plt.show()

'''
TASK 6
'''
qs = np.arange(0,5,10/n)
plt.plot(qs, task6_LHS)
plt.plot(qs, qs, 'r')
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Fixed-point quadrature scheme")
plt.show()

'''
TASK 7
'''
#Solve by the secant method
#first, interpolate 100 new points between each set of points for a total of 10000 points -> gives distance between points = 0.001
x7 = []
y7 = []

dq = 10/(n**2)

for i in range(int(n/2)-1):
    q = qs[i]
    x7.append(q)
    y0 = task7[i]
    y1 = task7[i+1]
    sl = (y1-y0)*(n/10)
    y7.append(y0)

    for j in range(1,n):
        x7.append(q+j*dq)
        y7.append(y0+j*dq*sl)

plt.plot(np.array(x7), np.array(y7))
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.show()

def secant(r0, r1, e):
    #Finds the closest value in x7 to r, assigns r1 that value
    if r0 in x7:
        r0_ind = x7.index(r0)
    else: 
        r0_ind = min(enumerate(x7), key=lambda x: abs(r0 - x[1]))[0]

    if r1 in x7:
        r1_ind = x7.index(r1)
    else: 
        r1_ind = min(enumerate(x7), key=lambda x: abs(r1 - x[1]))[0]

    fr0 = y7[r0_ind]
    fr1 = y7[r1_ind]

    r = r1 - fr1*(r1-r0)/(fr1-fr0)

    while abs(r-r1) > e:
        r0 = r1
        if r in x7:
            r1 = r
        else:
            r1 = min(enumerate(x7), key=lambda x: abs(r - x[1]))[1]
        

        if r0 in x7:
            r0_ind = x7.index(r0)
        else: 
            r0_ind = min(enumerate(x7), key=lambda x: abs(r0 - x[1]))[0]

        if r1 in x7:
            r1_ind = x7.index(r1)
        else: 
            r1_ind = min(enumerate(x7), key=lambda x: abs(r1 - x[1]))[0]

        fr0 = y7[r0_ind]
        fr1 = y7[r1_ind]

        r = r1 - fr1*(r1-r0)/(fr1-fr0)

    return r

print(secant(4, 4.1, 0.001))

'''
q1 = 4.299218368349152
'''
