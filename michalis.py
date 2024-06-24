import numpy as np


def michaelis_menten(x, alpha, lam):
    return alpha * x / (lam + x)



def curve_example(alpha, lam):

    linear_x = np.linspace(0, 1000, 100)

    nonlinear_y = michaelis_menten(linear_x, alpha, lam)
    
    return linear_x, nonlinear_y