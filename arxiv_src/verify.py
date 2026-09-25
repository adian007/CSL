import numpy as np
from scipy.integrate import quad
from math import sqrt, pi, exp

# 1) k-free double integral, uniform sphere, kernel exp(-s^2/(4 rC^2)), normalized by (int rho)^2
# eta = (1/V^2) * int_0^{2R} 4 pi s^2 exp(-s^2/(4 rC^2)) Vov(s) ds, Vov = pi/12 (4R+s)(2R-s)^2

def eta_kfree(s):  # s = R/rC
    R = s; rC = 1.0
    V = 4*pi*R**3/3
    def Vov(x):
        return pi/12*(4*R+x)*(2*R-x)**2
    def f(x):
        return 4*pi*x*x*exp(-x*x/(4*rC*rC))*Vov(x)
    val,_ = quad(f, 0, 2*R, limit=400)
    return val/V**2

print('k-free eta, prefactor eta/(rC/R)^3 :')
for s in [0.1, 1, 10, 100, 1000, 10000]:
    e = eta_kfree(s)
    print(f'  R/rC={s:8g}  eta={e:.6g}  eta/(1/s)^3={e*s**3:.6f}')
print('  6*sqrt(pi) =', 6*sqrt(pi))

# small-s check: eta -> 1
for s in [1e-2, 1e-3]:
    print(f'  small: R/rC={s}  eta={eta_kfree(s):.8f}')

# 2) NHH exact alpha_sphere, ratio to point value
# alpha/(m/amu)^2 = [exp(-R^2/r^2) -1 + R^2/(2r^2)(exp(-R^2/r^2)+1)] * 6 r^6/R^6

def alpha_ratio(s):  # s=R/rC
    x = s*s
    br = exp(-x) - 1 + x/2*(exp(-x)+1)
    return br*6/s**6

print('\nNHH alpha_sphere / (m/amu)^2 :')
for s in [1e-3, 0.1, 1, 10, 100]:
    a = alpha_ratio(s)
    print(f'  R/rC={s:8g}  alpha/m^2={a:.8g}  eta_NHH=alpha/(m^2/2)={2*a:.8g}  6*(rC/R)^4={6/s**4:.8g}')

# 3) Fourier form: eta = (4/sqrt(pi)) int q^2 e^{-q^2} |F_sph(q s)|^2 dq, F_sph(u)=3(sin u-u cos u)/u^3

def Fsph(u):
    if abs(u) < 1e-6:
        return 1.0 - u*u/10.0
    return 3*(sin(u)-u*cos(u))/u**3
from math import sin, cos

def eta_fourier(s):
    def f(q):
        u = q*s
        return q*q*exp(-q*q)*Fsph(u)**2
    val,_ = quad(f, 0, np.inf, limit=800)
    return 4/sqrt(pi)*val

print('\nfourier-form check:')
for s in [1, 10, 100]:
    e = eta_fourier(s)
    print(f'  R/rC={s:8g}  eta={e:.6g}  eta*s^3={e*s**3:.6f}')
