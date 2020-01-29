
# pylint: disable=relative-beyond-top-level

"""Implement some kernels for use with Gaussian Process."""

from copy import copy
import numpy as np
from scipy.optimize import minimize

from .gaussian_process import kernel_rbf, kernel_rbf_complex, kernel_rbf_complex_proper, compute_loglikelihood, compute_loglikelihood_complex


class Kernel:
    """Base class for Kernels."""

    def __init__(self, params, work_with_complex: float = False):
        # This will NOT be changed by optimize method
        self._initial_params = copy(params)
        # This WILL be changed by optimize method
        self._params = copy(params)
        self._work_with_complex_numbers = work_with_complex

    def compute_loglikelihood(self, x_train, y_train, noise_power):
        """Compute the likelihood of the kernel current parameters"""
        f = compute_loglikelihood_complex if self._work_with_complex_numbers else compute_loglikelihood
        return f(x_train, y_train, noise_power, kernel=self.compute, theta=self._params)

    def get_initial_params(self):
        """
        Get the kernel initial parameters.

        These are the parameters passed to the kernel during creation.
        """
        return self._initial_params

    def get_params(self):
        """Get the kernel parameters"""
        return self._params

    @property
    def work_with_complex_numbers(self):
        """True if this kernel can be used with complex numbers (inputs and target)"""
        return self._work_with_complex_numbers

    @staticmethod
    def compute(X1, X2, *params):
        """Compute the similarity between `X1``and `X2` according with the kernel."""
        raise NotImplementedError("Implement-me")

    def __call__(self, X1, X2):
        return self.compute(X1, X2, *self._params)

    def optimize(self, x_train, y_train, noise_power: float):
        """Optimize the kernel parameters"""
        f = compute_loglikelihood_complex if self._work_with_complex_numbers else compute_loglikelihood

        def function_to_optimize(theta):
            num_samples = x_train.shape[0]
            return -f(x_train, y_train, noise_power, kernel=self.compute, theta=theta) / num_samples

        # See https://stackoverflow.com/questions/19244527/scipy-optimize-how-to-restrict-argument-values
        # for option to set bounds for the optimized parameters
        res = minimize(function_to_optimize,
                       self._initial_params, method="Nelder-Mead")
        self._params = res.x

    def clone(self):
        """Create a new kernel with the same parameters of this one"""
        return self.__class__(*self._initial_params)


class RBF(Kernel):
    """
    Isotropic squared exponential kernel.

    Parameters
    ----------
    l : float
        The length scale of of the RBF kernel.
    sigma_f : float
        The constant value multiplying the complex exponential.
    """

    def __init__(self, l: float, sigma_f: float):
        super().__init__(params=[
            l, sigma_f], work_with_complex=False)

    @staticmethod
    def compute(X1, X2, *params):
        """Compute the similarity between `X1``and `X2` according with the kernel."""
        return kernel_rbf(X1, X2, *params)


class RBF_Complex(Kernel):
    """
    Isotropic squared exponential kernel for complex case.
    """

    def __init__(self, l: float, sigma_f: float):
        super().__init__(params=[
            l, sigma_f], work_with_complex=True)

    @staticmethod
    def compute(X1, X2, *params):
        """Compute the similarity between `X1``and `X2` according with the kernel."""
        return kernel_rbf_complex(X1, X2, *params)


class RBF_ComplexProper(Kernel):
    """
    Isotropic squared exponential kernel for complex proper case.
    """

    def __init__(self, l: float, sigma_f: float):
        super().__init__(params=[
            l, sigma_f], work_with_complex=True)

    @staticmethod
    def compute(X1, X2, *params):
        """Compute the similarity between `X1``and `X2` according with the kernel."""
        return kernel_rbf_complex_proper(X1, X2, *params)


if __name__ == "__main__":
    import math

    num_samples = 40
    num_features = 4
    noise_power = 1e-4
    rbf = RBF(1.0, 1.0)

    X1 = np.random.randn(num_samples, num_features)
    print(rbf.get_params())
    K = rbf(X1, X1)
    print(K.shape)
    with np.printoptions(linewidth=140, precision=2):
        print(K)
    print(rbf.work_with_complex_numbers)

    y1 = X1 @ [1, -2, 3, -4] + \
        math.sqrt(noise_power) * np.random.randn(num_samples)

    rbf.optimize(X1, y1, noise_power)

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    rbf2 = rbf.clone()
    print(rbf is rbf2)
    print("rbf:", rbf.get_initial_params())
    print("rbf:", rbf.get_params())
    print("rbf2:", rbf2.get_initial_params())
    print("rbf2:", rbf2.get_params())
