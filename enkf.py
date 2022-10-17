import numpy as np


class EnsembleKalmanFilter:
    def __init__(self):
        """
        Ensemble Kalman Filter (EnKF)

        EnKF following the formulation found in Iglesias et al. (2013),
        The Ensemble Kalman Filter for Inverse Problems.
        doi:10.1088/0266-5611/29/4/045001

        :param maxit: int, maximum number of iterations
        :param n_batches, int,  number of batches to used in mini-batch. If set
            to `1` uses the whole given dataset. Default is `1`.
        :param online, bool, True if one random data point is requested,
            between [0, dims], otherwise do mini-batch. `dims` is the number of
            observations. Default is False
        """
        self.Cpp = None
        self.Cup = None
        self.ensemble = None
        self.observations = None

        self.gamma = 0.
        self.gamma_s = 0
        self.dims = 0
        self.cov_mat = {'Cpp': [], 'Cup': []}

    def fit(self, ensemble, ensemble_size, observations, model_output, gamma):
        """
        Prediction and update step of the EnKF
        Calculates new ensembles.

        :param ensemble: nd numpy array, contains ensembles `u`
        :param ensemble_size: int, number of ensembles
        :param observations: nd numpy array, observation or targets
        :param model_output: nd numpy array, output of the model
            In terms of the Kalman Filter the model maps the ensembles (dim n)
            into the observed data `y` (dim k).
        :param  gamma: nd numpy array, Normalizes the model-data distance in the
            update step, :`noise * I` (I is identity matrix) or
            :math:`\\gamma=I` if `noise` is zero
        :return self, Possible outputs are:
            ensembles: nd numpy array, optimized `ensembles`
            Cpp: nd numpy array, covariance matrix of the model output
            Cup: nd numpy array, covariance matrix of the model output and the
                ensembles
        Notes:
            Dimension are:
            ensemble member dim d
            model output dim kxd
            D(U) = Cpp dim kxk
            C(U) = Cup dim kxd
        """
        # copy the data so we do not overwrite the original arguments
        self.ensemble = ensemble
        self.observations = observations
        # convert to pytorch
        self.ensemble = np.array(self.ensemble)
        self.observations = np.array(self.observations)
        self.gamma = np.array(gamma)
        model_output = np.array(model_output)
        # Calculate the covariances
        self.Cpp = _cov_mat(model_output, model_output, ensemble_size)
        self.Cup = _cov_mat(self.ensemble, model_output.T, ensemble_size)
        self.ensemble = _update_step(self.ensemble,
                                     observations,
                                     model_output, self.gamma,
                                     self.Cpp, self.Cup)
        self.cov_mat['Cpp'].append(self.Cpp)
        self.cov_mat['Cup'].append(self.Cup)
        return self

    def clear(self):
        self.gamma = []
        self.ensemble = []
        self.Cpp = []
        self.Cup = []


def _update_step(ensemble, observations, g, gamma, Cpp, Cup):
    """
    Update step of the kalman filter
    Calculates the covariances and returns new ensembles
    """
    obs = (observations - g)
    cpp_gamma = Cpp + gamma
    s = np.linalg.solve(cpp_gamma, obs.T)
    print(f's {s.mean()}')
    r = Cup @ s.T
    print(f'r {r.mean()}')
    return r + ensemble
    # return (Cup @ np.linalg.solve(Cpp+gamma, (observations-g).T)).T + ensemble


def _cov_mat(x, y, ensemble_size):
    """
    Covariance matrix
    """
    return np.tensordot((x - x.mean(0)), (y - y.mean(0)),
                        axes=0) / ensemble_size