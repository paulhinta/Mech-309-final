from numpy import cos, cosh, sin, sinh
import numpy as np
import matplotlib.pyplot as plt

import time
'''
TASK 9
'''
r1 = 1.87527632324985

#r1 and q1 derived in previous task
#first, we must resolve for q1; each value of FN will dictate what q1 is
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

qs = np.arange(0, 5, (5-0)/n)               #100 data points for now
xs = np.arange(0, 1, (1-0)/n)                #100 data points

first_integral = []

# print(qs)
# print(xs)

def integrand(x0, q0, s0):
    return phi(x0)*cos(q0*(phi(x0) - phi(s0)))

print("Solving the first integral")
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

print("Solving the second integral at different FN")
dq = 5/n
for f in range(0,n+1,1):
    j=0
    fs = []
    for q in range(n):
        q0 = j*dq

        lst = first_integral[q]
        dx = 1/n
        total = lst[0] + lst[n-1]

        for i in range(1, n-2):
            total += lst[i]*2

        out = f*(total*dx/2)

        fs.append(out + y*q0)

        j+=1

    second_integral.append(fs)

# print(q)
# for f in second_integral:
#     print(f)

def secant(x7, y7, r0, r1, e):                  #x7 and y7 are the x- and y-axes
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

    steps = 0

    while abs(r-r1) > e:
        print("r0: " + str(r0))
        print("r1: " + str(r1))
        print("r: " + str(r))
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

        if fr0==fr1:
            return r

        r = r1 - fr1*(r1-r0)/(fr1-fr0)

        steps += 1
        if steps > 20:
            return r

    return r

roots = []

print("Now finding the roots q1 for different FN")
for f in second_integral:
    # dq = 10/(n**2)

    x_f = []
    y_f = f

    for q in qs:
        x_f.append(q)

    # for i in range(n-1):
    #     q = qs[i]
    #     x_f.append(q)
    #     y0 = f[i]
    #     y1 = f[i+1]
    #     sl = (y1-y0)*(n/10)
    #     y_f.append(y0)

    #     for j in range(1,n):
    #         x_f.append(q+j*dq)
    #         y_f.append(y0+j*dq*sl)
    
    r = secant(x_f, y_f, 0, 0.10, 0.001)
    roots.append(r)

fs = np.arange(0, n+1, 1)

print(second_integral[1])
print(roots)

#sub x = L

trans_deflection = []

def psi(q1):
    return q1*(sin(r1*1) + sinh(r1*1) + ((cos(r1)+cosh(r1))/(sin(r1)+sinh(r1)))*(cos(r1*1)-cosh(r1*1)))

for i in range(len(fs)):
    trans_deflection.append(psi(roots[i]))

plt.plot(fs, np.array(trans_deflection))
plt.xlabel("Magnitude of the External Force FN")
plt.ylabel("Transverse deflection of the beam at x=L, w(L) [m]")
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.title("Transverse deflection of the beam at the end vs Magnitude of External Force")
plt.show()

# y = [0.4183457811259088, 0.2424556628888618, 0.06638700431977829, -0.10985974429618689, -0.28628383372943433, -0.46288421767047006, -0.6396595553323853, -0.8166082147834278, -0.9937282769996163, -1.1710175406251884, -1.3484735274265516, -1.5260934884233344, -1.703874410678117, -1.8818130247244587, -2.0599058126109715, -2.2381490165373377, -2.4165386480564983, -2.5950704978155255, -2.773740145806201, -2.952542972094828, -3.131474167999463, -3.310528747681522, -3.489701560117538, -3.668987301415886, -3.8483805274423197, -4.0278756667174225, -4.207467033548404, -4.387148841357107, -4.566915216165709, -4.746760210201304, -4.926677815580364, -5.106661978034081, -5.286706610635656, -5.466805607490835, -5.646952857353313, -5.8271422571271305, -6.007367725218713, -6.187623214701964, -6.367902726260619, -6.5482003208729775, -6.728510132205203, -6.908826378680506, -7.089143375192751, -7.269455544434412, -7.449757427810171, -7.630043695908986, -7.810309158509086, -7.990548774091913, -8.170757658842884, -8.350931095118513, -8.53106453936137, -8.711153629446173, -8.891194191442281, -9.071182245779827, -9.25111401280863, -9.430985917741166, -9.610794594972791, -9.790536891774456, -9.970209871355232, -10.149810815293884, -10.329337225340808, -10.50878682459358, -10.688157558051298, -10.867447592554873, -11.046655316122182, -11.225779336688934, -11.40481848026775, -11.583771788539748, -11.76263851589454, -11.941418125936025, -12.120110287473032, -12.298714870015097, -12.47723193879507, -12.655661749341524, -12.83400474162496, -13.012261533803002, -13.190432915590591, -13.36851984128215, -13.546523422453385, -13.72444492037101, -13.902285738139343, -14.080047412612977, -14.257731606105239, -14.435340097922236, -14.612874775752523, -14.790337626942309, -14.96773072968615, -15.145056244162813, -15.322316403645706, -15.499513505616918, -15.67664990291336, -15.853727994932969, -16.030750218928247, -16.207719041413622, -16.384636949712355, -16.56150644366767, -16.738330027541956, -16.915110202126616, -17.0918494570843, -17.26855026354371]
# lst = []
# for q in qs:
#     lst.append(q)

# x = secant(qs, y, 0, 0.1, 0.01)

# print(x)