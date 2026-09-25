import numpy as np
from scipy.integrate import quad
from math import sqrt, pi, exp
from scipy.special import erf

# closed form: eta = 3/(16 R^6) * [16 R^3 A2 - 12 R^2 A3 + A5], beta=1/(4 rC^2), c=2R
def eta_closed(s):  # s = R/rC, set rC=1
    R = s; b = 1.0/4.0; c = 2*R; U = b*c*c  # = R^2
    A2 = (1/(4*b))*(sqrt(pi/b)*erf(c*sqrt(b)) - 2*c*exp(-b*c*c))
    A3 = (1/(2*b*b))*(1 - exp(-b*c*c)*(1+b*c*c))
    A5 = (1/(2*b**3))*(2 - exp(-U)*(U*U+2*U+2))
    return 3.0/(16*R**6) * (16*R**3*A2 - 12*R**2*A3 + A5)

# compare against direct quadrature (from before)
def eta_num(s):
    R=s
    V=4*pi*R**3/3
    def Vov(x): return pi/12*(4*R+x)*(2*R-x)**2
    val,_=quad(lambda x: 4*pi*x*x*exp(-x*x/4)*Vov(x), 0, 2*R, limit=500)
    return val/V**2

for s in [0.5, 1, 10, 100, 1000]:
    e1=eta_closed(s); e2=eta_num(s)
    print(f's={s:7g}  closed={e1:.8g}  quad={e2:.8g}  ratio={e1/e2 if e2 else 1:.10f}')
print('6*sqrt(pi)=',6*sqrt(pi), ' 3/sqrt(pi)=',3/sqrt(pi))
