####  Set of routines to advance the Lorenz 63 model using 4th order Runge-Kutta scheme.

import numpy as np

#  Compute right-hand side of the Lorenz 63 model equations
def lorenz(x, y, z, s=10., rho=28., b=(8./3.)):
    return s*(y - x), (rho-z)*x - y, x*y - b*z

def advance(xin, yin, zin, tadv, dt=0.001):

    num_steps = int(round(tadv / dt))

    x = xin
    y = yin
    z = zin

    for i in range(num_steps):

       xrhs1, yrhs1, zrhs1 = lorenz(x, y, z)

       xtild = x + xrhs1 * dt * 0.5 
       ytild = y + yrhs1 * dt * 0.5
       ztild = z + zrhs1 * dt * 0.5
       xrhs2, yrhs2, zrhs2 = lorenz(xtild, ytild, ztild)

       xtild = x + xrhs2 * dt * 0.5
       ytild = y + yrhs2 * dt * 0.5
       ztild = z + zrhs2 * dt * 0.5
       xrhs3, yrhs3, zrhs3 = lorenz(xtild, ytild, ztild)

       xtild = x + xrhs3 * dt
       ytild = y + yrhs3 * dt
       ztild = z + zrhs3 * dt 
       xrhs4, yrhs4, zrhs4 = lorenz(xtild, ytild, ztild)

       x = x + dt * (xrhs1 + 2.0*xrhs2 + 2.0*xrhs3 + xrhs4) / 6.0
       y = y + dt * (yrhs1 + 2.0*yrhs2 + 2.0*yrhs3 + yrhs4) / 6.0
       z = z + dt * (zrhs1 + 2.0*zrhs2 + 2.0*zrhs3 + zrhs4) / 6.0

    return x, y, z

#  Compute the tangent linear model matrix for Lorenz 63 model by brute force 
#  based on input forecast trajectory
def calc_tlm(tvec, xvec, yvec, zvec, s=10., rho=28., b=(8./3.)):

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])

    for t in range(len(tvec)-1):

      tlm = np.array([[-s, s, 0], [(rho-zvec[t]), -1., -xvec[t]], [yvec[t], xvec[t], -b]])

      v1 = v1 + np.matmul(tlm,v1)*(tvec[t+1]-tvec[t])
      v2 = v2 + np.matmul(tlm,v2)*(tvec[t+1]-tvec[t])
      v3 = v3 + np.matmul(tlm,v3)*(tvec[t+1]-tvec[t])

    M = np.transpose([v1, v2, v3])

    return M

