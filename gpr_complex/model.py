# pylint: disable=relative-beyond-top-level
"""
Models for Gaussian Process.

Currently there is only GPR, for Gaussian Process Regression.
"""

import numpy as np
from bokeh.models import Label
from .kernels import Kernel, RBF
from .gaussian_process import posterior_predictive
from .plot import GpChart


class GPR:
    """
    Class for Gaussian Process regressor

    Parameters
    ----------
    noise_power : float
        The noise power.
    kernel : Kernel
        The kernel to use for the GP model.
    """
    def __init__(self, noise_power: float, kernel: Kernel):
        self._kernel = kernel.clone()
        self._noise_power = noise_power

        # This will be set by the `fit` method, which wil also update params
        self._x_train = None
        self._y_train = None
        self._likelihood = None

    @property
    def kernel(self):
        """Return the kernel used by this GPR model."""
        return self._kernel

    def fit(self, x_train, y_train):
        """
        Fit the Gaussian process model.

        Parameters
        ----------
        x_train : np.ndarray
            The input. Dimension: num_samples x num_features
        y_train : np.ndarray
            The variable to be predicted. Dimension: num_samples
        """
        self._x_train = np.copy(x_train)
        self._y_train = np.copy(y_train)
        self._kernel.optimize(x_train, y_train, self._noise_power)
        self._likelihood = self._kernel.compute_loglikelihood(
            x_train, y_train, self._noise_power)

    @property
    def input_dim(self):
        """True if the model was already trained with the `fit` method, False otherwise."""
        if self._x_train is None:
            return 0
        return self._x_train.shape[1]

    @property
    def is_trained(self):
        """Property indicating if the model is already trained or not"""
        return self._x_train is not None

    def likelihood(self):
        """
        Get the likelihood of the trained kernel.

        It returns None if the model was not trained yet.
        """
        return self._likelihood

    def predict(self, x_test, return_cov=False):
        """
        Predict the target for the provided input `x_test` using the fitted
        model.

        Parameters
        ----------
        x_test : np.ndarray The test samples. Dimension: `num_samples` x
            `num_features`

        Returns
        -------
        np.ndarray, np.ndarray Posterior mean vector (with `n` samples) and
            covariance matrix (dimension `n x n`).
        """
        if not self.is_trained:
            raise RuntimeError(
                "The model needs to be trained with the `fit` method before it can make predictions"
            )
        mu, cov = posterior_predictive(x_test, self._x_train, self._y_train,
                                       self._noise_power, self._kernel.compute,
                                       self._kernel.get_params())
        if return_cov:
            return mu, cov
        return mu

    def _get_chart_real_case(self, X):
        mu, cov = self.predict(X)
        chart = GpChart(dict(mu=mu.ravel(), cov=cov, x=X.ravel()),
                        dict(x=self._x_train.ravel(), y=self._y_train.ravel()),
                        dont_show=True)
        loglikelihood_value = self.kernel.compute_loglikelihood(
            self._x_train, self._y_train, self._noise_power)
        text_annotation = Label(
            x=10,
            y=300,
            text=f'Log Likelihood Value: {loglikelihood_value:.4}',
            x_units="screen",
            y_units="screen")
        chart.fig.add_layout(text_annotation)
        return chart

    def chart(self, X, show=True):
        """Show a chart with the GP prediction for `X`.

        This is only possible when `X` has only one feature and it is real.
        """
        if self.input_dim != 1:
            raise RuntimeError(
                "The chart method can only be used when the model is trained with 1D inputs"
            )

        if X.dtype == complex:
            raise RuntimeError(
                "Chart method does not work for inputs in the complex field")

        chart = self._get_chart_real_case(X)
        if show:
            chart.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    import math

    num_samples = 400
    num_features = 4
    noise_power = 1e-5
    rbf = RBF(1.0, 1.0)

    X1 = np.random.randn(num_samples, num_features)
    y1 = X1 @ [1, -2, 3, -4] + \
        math.sqrt(noise_power) * np.random.randn(num_samples)

    gpr = GPR(noise_power, rbf)
    print(gpr.kernel.get_params())
    gpr.fit(X1, y1)
    print(gpr.kernel.get_params())

    y1_pred, y1_pred_cov = gpr.predict(X1)
    print("y1", y1.shape)
    print("y1_pred", y1_pred.shape)
    print(y1 - y1_pred)
