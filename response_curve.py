from michalis import michaelis_menten
from scipy.optimize import curve_fit


def response_curves(treatment_effect, df_scoring):
    
    maxfev = 100000
    lam_initial_estimate = 0.001
    alpha_initial_estimate = max(treatment_effect)
    initial_guess = [alpha_initial_estimate, lam_initial_estimate]

    popt, pcov = curve_fit(michaelis_menten, df_scoring['T'], treatment_effect, p0=initial_guess, maxfev=maxfev)
    
    return popt, pcov