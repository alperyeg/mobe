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
    ensemble_size = 20
    observation_size = 10
    y = np.zeros(observation_size)
    init_ensemble = rng.uniform(-1, 1, size=(ensemble_size, observation_size))
    # init class
    enkf = EnsembleKalmanFilter()
    # hyperparameter, scaling factor
    gamma_scale = 1.
    iterations = 100
    gamma = (np.eye(observation_size) * gamma_scale)
    lambda1 = np.linspace(0, 1, 25, endpoint=True)
    lambda2 = 1 - lambda1
    results = []
    for k, l in zip(lambda1, lambda2):
        model_output = k * model1(init_ensemble) + l * model2(init_ensemble)
        ensemble = init_ensemble.copy()
        print(k, l)
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
            # print(ensemble.shape)
            # print(ensemble.mean())
            if i == iterations - 1:
                results.append(ensemble)
    results = np.array(results)
    # x-axis  norm(G1 - y)
    # y-axis norm(G2 - y)**2
    xs = []
    ys = []
    for idx, r in enumerate(results):
        x = np.linalg.norm(model1(r.mean()))
        print(f'r mean {model1(r.mean())} r norm {x}')
        y_i = model2(r.mean())
        # print(f'r mean {r.mean()} r norm {y_i}')
        xs.append(x)
        ys.append(y_i)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(xs, ys, '.')
    plt.show()
