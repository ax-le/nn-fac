import numpy as np
from scipy.special import lambertw # The lambertw function is imported from the scipy.special module.

eps = 1e-12

def deep_KL_mu(W_Lm1, W_L, H_L, WH_Lp1, lambda_):
    ONES = np.ones_like(W_Lm1)
    a = ONES @ H_L.T - lambda_ * np.log(WH_Lp1)
    b = W_L * ((W_Lm1 / (W_L @ H_L)) @ H_L.T)
    lambert = lambertw(b * np.exp(a/lambda_) / lambda_, k=0).real
    W_L = np.maximum(eps, (1/lambda_ * b) / (lambert + eps))

    return W_L

