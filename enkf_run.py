import numpy as np
import matplotlib.pyplot as plt

from enkf import EnsembleKalmanFilter
from numpy.random import default_rng


'''
Main script to run the EnKF calculation.
Similar style to scikit, the EnKF class is initialized and
fit is the method to obtain the new ensemble.

'''
def model1(x):
    return (x - 0.5)**2

def model2(x):
    return (x + 0.5)**2


if __name__ == '__main__':
    rng = default_rng(seed=0)
    # observations or targets
    y = np.zeros(20)
    ensemble_size = 20
    init_ensemble = rng.uniform(-1, 1, size=ensemble_size)
    # init class
    enkf = EnsembleKalmanFilter()
    # hyperparameter, scaling factor
    gamma_scale = 1.
    iterations = 1
    gamma = (np.eye(len(y)) * gamma_scale)
    lambda1 = np.arange(0, 1.1, 0.1)
    lambda2 = 1 - lambda1
    results = []
    # model_output = lambda1.reshape(-1, 1) * model1(ensemble) + lambda2.reshape(-1, 1) * model2(ensemble)
    for k, l in zip(lambda1, lambda2):
        model_output = k * model1(init_ensemble) + l * model2(init_ensemble)
        ensemble = init_ensemble.copy()
        for i in range(iterations):
            # calculate the new ensemble
            enkf.fit(ensemble=ensemble,
                     ensemble_size=ensemble_size,
                     observations=y,
                     model_output=model_output,
                     gamma=gamma
                     )
            # access the new members
            ensemble = enkf.ensemble
            print(ensemble.shape)
            print(ensemble.mean())
            results.append(ensemble)
    # plot histogram
    for idx, r in enumerate(results):
        plt.hist(r.ravel(), label=str(i))
        plt.legend()
    plt.show()
