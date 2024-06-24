import numpy as np
from michalis import michaelis_menten


def data_generator(n, tau_weight, alpha, lam):

    # Set number of features
    p=4

    # Create features
    X = np.random.uniform(size=n * p).reshape((n, -1))

    # Nuisance parameters
    b = (
        np.sin(np.pi * X[:, 0])
        + 2 * (X[:, 1] - 0.5) ** 2
        + X[:, 2] * X[:, 3]
    )

    # Create treatment and treatment effect
    T = np.linspace(200, 10000, n)
    T_mm = michaelis_menten(T, alpha, lam) * tau_weight
    tau = T_mm / T

    # Calculate outcome
    y = b + T * tau + np.random.normal(size=n) * 0.5
    
    y_train = y
    X_train = np.hstack((X, T.reshape(-1, 1)))
    
    return y_train, X_train, T_mm, tau