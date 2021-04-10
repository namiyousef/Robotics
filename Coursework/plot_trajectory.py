import plac
import numpy as np
import math as m
# TODO later, the number wil have to be replaced with sympy code
@plac.annotations(
    p0=('Initial coordinates of manipulator', 'positional', None, str),
    pf=('Final coordinates of manipulator', 'positional', None, str),
    t=('Time for the movement', 'positional', None, float),
    steps=('Number of steps to make the movement','positional', None, int),
    # TODO make sure via points designed properly
    via=('Via points', 'option', None, list)
)

# TODO in reality this would have to interface with the symbolic stuff
manipulator_params = {
    'Le' :1,
    'L3' :1,
    'L1' :1,
    'phi':m.pi/4,
}

def main(p0, pf, td, steps, via = None):
    p0 = list(map(float, p0.strip('[]').split(',')))
    pf = list(map(float, pf.strip('[]').split(',')))
    p0 = np.asarray(p0 + [1]).reshape(-1,1)
    pf = np.asarray(pf + [1]).reshape(-1,1)



def check_in_workspace(point):
    pass

def cubic_interpolator(q0, qf, t, td):
    a0 = q0
    a1 = 0
    a2 = (3/(td**2))*(qf - q0)
    a3 = -(2/(td**3))*(qf - q0)
    qi = a0 + a1*t + a2*t**2 + a3*t**3
    qdoti = a1 + 2*a2*t + 3*a3*t**2
    qddoti = 2*a2 + 6*a3*t
    return qi, qdoti, qddoti

def find_theta1(x,y,L1, Le, phi, d2):
    l = (L1+d2)*m.cos(phi)
    theta1 = m.atan2(Le*y - l*x, Le*x + l*x)
    pass

def find_d2():
    pass

def find_theta3():
    pass

if __name__ == '__main__':
    plac.call(main)