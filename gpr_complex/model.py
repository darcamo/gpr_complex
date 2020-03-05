# pylint: disable=relative-beyond-top-level
"""
Models for Gaussian Process.

Currently there is only GPR, for Gaussian Process Regression.
"""

from typing import Optional, Tuple, Union, cast

import numpy as np
from bokeh.models import Label

from .gaussian_process import posterior_predictive
from .kernels import Kernel
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
        self._x_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._likelihood: Optional[float] = None

    @property
    def kernel(self) -> Kernel:
        """Return the kernel used by this GPR model."""
        return self._kernel

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
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

    def refit(self) -> None:
        """
        Refit the model.

        Calling this method only makes sense if you have added new data without
        fitting the model with the `add_new_data` method.
        """
        self._kernel.optimize(self._x_train, self._y_train, self._noise_power)
        self._likelihood = self._kernel.compute_loglikelihood(
            self._x_train, self._y_train, self._noise_power)

    @property
    def input_dim(self) -> int:
        """
        Number of dimensions features in the training sample
        """
        if self._x_train is None:
            return 0
        return cast(int, self._x_train.shape[1])

    @property
    def is_trained(self) -> bool:
        """Property indicating if the model is already trained or not"""
        return self._x_train is not None

    @property
    def likelihood(self) -> Optional[float]:
        """
        Get the likelihood of the trained kernel.

        It returns None if the model was not trained yet.
        """
        return self._likelihood

    def add_new_data(self,
                     x_new: np.ndarray,
                     y_new: np.ndarray,
                     discard_oldest_entries: bool = False) -> None:
        """
        Add more data to the model without actually training the model again.

        Parameters
        ----------
        x_new : np.ndarray
            The new input data.
        y_new : np.ndarray
            The new target data.
        """
        assert self._x_train is not None
        assert self._y_train is not None

        if discard_oldest_entries:
            num_entries = x_new.shape[0]
            self._x_train = np.concatenate(
                [self._x_train[num_entries:], x_new])
            self._y_train = np.concatenate(
                [self._y_train[num_entries:], y_new])
        else:
            self._x_train = np.concatenate([self._x_train, x_new])
            self._y_train = np.concatenate([self._y_train, y_new])

    def predict(
        self,
        x_test: np.ndarray,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict the target for the provided input `x_test` using the fitted
        model.

        Parameters
        ----------
        x_test : np.ndarray
            The test samples. Dimension: `num_samples` x `num_features`
        return_cov : bool
            If set to True, the covariance matrix of the predictive posterior is
            also returned.

        Returns
        -------
        np.ndarray or (np.ndarray, np.ndarray)
            Posterior mean vector (with `n` samples) and covariance matrix
            (dimension `n x n`).
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

    def predict_and_add_new_data(self,
                                 x_test: np.ndarray,
                                 y_test: np.ndarray,
                                 discard_oldest_entries: bool = False,
                                 refit: bool = False) -> np.ndarray:
        """
        Predict the targets for the provided input `x_test` using the fitted model.

        After predicting the target for each new entry in `x_test` the entry and
        corresponding true target in `y_test` are added as new data (see
        `add_new_data` method) without re-training the model. This is suitable
        when the model is used to perform prediction on a series and after the
        prediction is performed the true value will be known, which can then be
        used for future predictions.

        Parameters
        ----------
        x_test : np.ndarray
            The test samples. Dimension: `num_samples` x `num_features`
        y_test : np.ndarray
            The variable to be predicted. Dimension: num_samples
        refit : bool
            If True, refit the model after each new entry

        Returns
        -------
        np.ndarray
            The prediction.
        """
        assert (y_test.ndim == 1)
        y_pred = np.empty_like(y_test)
        for i in range(y_test.size):
            cur_x = x_test[i:i + 1]
            y_pred[i] = self.predict(cur_x).item()  # type: ignore
            # Add the new data
            self.add_new_data(cur_x, y_test[i:i + 1], discard_oldest_entries)
            if refit:
                # Fit the model again with the new data
                self.refit()
        return y_pred

    def _get_chart_real_case(self, X: np.ndarray) -> GpChart:
        assert (self._x_train is not None)
        assert (self._y_train is not None)

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

    def chart(self, X: np.ndarray, show: bool = True) -> None:
        """
        Show a chart with the GP prediction for `X`.

        This is only possible when `X` has only one feature and it is real.

        Parameters
        ----------
        X : np.ndarray
            The array to with the inputs to prediction the output
        show : bool, optional
            If the plot should be shown. Default is True.

        Raises
        ------
        RuntimeError
            If the model was trained with input that is not 1D or if the input is complex.
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
