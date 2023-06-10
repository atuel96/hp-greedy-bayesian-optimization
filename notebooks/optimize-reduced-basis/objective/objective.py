import numpy as np
from skreducedmodel.reducedbasis import ReducedBasis, error


class Objective:
    """Callable objective class to perform an optimization within an optuna study"""

    VALID_HYPERPARAMS = [
        "q_index",
        "chi_index",
        "chi1_index",
        "chi2_index",
        "greedy_tol",
        "nmax",
        "lmax",
    ]

    def __init__(
        self,
        times: np.ndarray,
        train: np.ndarray,
        valid: np.ndarray,
        parameters_train: np.ndarray,
        parameters_valid: np.ndarray,
        hyperparameters: dict,
    ):
        """Returns objective class to optimize

        Parameters
        ----------
            times: numpy.ndarray
            train: numpy.ndarray
            valid: numpy ndarray
            parameters_train : numpy.ndarray
            parameters_valid : numpy.ndarray
            hyperparameters : dict

        """
        self.times = times
        self.train = train
        self.valid = valid
        self.parameters_train = parameters_train
        self.parameters_valid = parameters_valid
        self.hyperparameters = hyperparameters

    def __call__(self, trial):
        trial_hyperparams = {}
        seed_trial_hyperparams = {}
        for key, value in self.hyperparameters.items():
            assert (
                key in self.VALID_HYPERPARAMS
            ), f"'{key}' is not a valid hyperparameter name. Try one of the following:\n{self.VALID_HYPERPARAMS}"

            if type(value) == list or type(value) == tuple:
                assert (
                    len(value) == 2
                ), "You must specify the range of values for the search space in a list or tuple [min_value, max_value]"

                if "index" in key:
                    seed_trial_hyperparams[key] = trial.suggest_int(
                        key, value[0], value[1]
                    )
                elif key == "greedy_tol":
                    trial_hyperparams[key] = trial.suggest_float(
                        key, value[0], value[1], log=True
                    )
                else:
                    trial_hyperparams[key] = trial.suggest_int(key, value[0], value[1])
                continue
            assert (
                type(value) == int or type(value) == float
            ), f"value of '{key}' must be int or float"
            trial_hyperparams[key] = value

        # global seed index
        seed_params_size = {
            k: v[1] + 1 for k, v in self.hyperparameters.items() if "index" in k
        }
        index_seed_global_rb = get_global_index(
            seed_params_size, seed_trial_hyperparams
        )

        # seed_index = q_index*chis_train.shape[0]**2 + chi1_index*chis_train.shape[0] + chi2_index

        # build a reduced basis
        rb = ReducedBasis(
            index_seed_global_rb=index_seed_global_rb, **trial_hyperparams
        )
        rb.fit(self.train, self.parameters_train, self.times)

        projections = []
        for h_t, q_t in zip(self.valid, self.parameters_valid):
            projections.append(rb.transform(h_t, q_t))
        projections = np.asarray(projections)
        errors = []

        for h_proy, h_valid in zip(projections, self.valid):
            errors.append(error(h_proy, h_valid, self.times))
        return max(errors)


def get_global_index(seed_params_size: dict, seed_trial_values: dict) -> int:
    """Gets global seed index from 'q' and 'chi' values. Order matters!

    Parameters
    ----------
    seed_params_size : dict
        Dictionary with seed parameter names and its size
    seed_trial_values : dict
        Dictionary with seed parameter names and its value

    Returns
    -------
    int
        index of global seed
    """

    assert (
        seed_params_size.keys() == seed_trial_values.keys()
    ), "both dicts must contain the same keys"

    seed_params = list(seed_params_size.keys())
    index_seed_global_rb = 0
    for i in range(len(seed_params_size) - 1):
        trial_value = seed_trial_values[seed_params[i]]
        param_sizes_product = 1
        for param_size in list(seed_params_size.values())[i + 1 :]:
            param_sizes_product *= param_size
        index_seed_global_rb += trial_value * param_sizes_product
    # last index
    index_seed_global_rb += seed_trial_values[seed_params[-1]]
    return index_seed_global_rb
