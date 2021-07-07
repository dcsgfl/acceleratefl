#
# dp.py
# Differential Privacy Routines
#

import numpy as np

#
# Add noise to a histogram via the Laplace mechanism.
# This preserves (epsilon, 0)-differential privacy.
#
def dp_add_hist_noise(histParm, epsilon):

    noiseArray = np.random.laplace(0, 1/epsilon, len(histParm))
    return histParm + noiseArray
