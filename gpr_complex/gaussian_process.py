"""
Module with several functions to work with Gaussian processes.
"""

import math

from typing import Optional, Callable, List, Union, Iterable, Tuple, cast
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

KernelFunc = Callable[..., np.ndarray]
DiagLoading = Union[float, bool]
ParamsType = List[float]  # Type for kernel parameters

# def kernel_rbf(X1, X2, l=1.0, sigma_f=1.0):
#     """
#     Isotropic squared exponential kernel. Computes a covariance matrix from
#     points in X1 and X2.

#     Parameters
#     ----------
#     X1: np.ndarray
#         A 2D array of `m` points (dimension is m x d).
#     X2: Array of n
#     points (n x d).

#     Returns
#     -------
#     np.ndarray
#         A 2D array corresponding to the covariance function.
#         Dimension is (m x n).
#     """

#     sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
#     return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def kernel_rbf(X1: np.ndarray,
               X2: np.ndarray,
               l: float = 1.0,
               sigma_f: float = 1.0) -> np.ndarray:
    """
    Isotropic squared exponential kernel. Computes a covariance matrix from
    points in X1 and X2.

    Parameters
    ----------
    X1: np.ndarray
        A 2D array of `m` points (dimension is m x d).
    X2: np.ndarray
        Array of n points (n x d).
    l : float
        The length-scale of the kernel
    sigma_f : float
        The standard deviation of the kernel

    Returns
    -------
    np.ndarray
        A 2D array corresponding to the covariance function.
        Dimension is (m x n).
    """
    sqdist = np.linalg.norm((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]),
                            axis=2)**2
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


def kernel_rbf_complex(X1: np.ndarray,
                       X2: np.ndarray,
                       l: float = 1.0,
                       sigma_f: float = 1.0) -> np.ndarray:
    """
    RBF implementation for complex case

    This version works with the improper complex case.

    Parameters
    ----------
    X1: np.ndarray
        A 2D array of `m` points (dimension is m x d).
    X2: np.ndarray
        Array of n points (n x d).
    l : float
        The length-scale of the kernel
    sigma_f : float
        The standard deviation of the kernel

    Returns
    -------
    np.ndarray
        A 2D array corresponding to the covariance function.
        Dimension is (m x n).
    """
    krr = kernel_rbf(X1.real, X2.real, l, sigma_f)
    kii = kernel_rbf(X1.imag, X2.imag, l, sigma_f)
    kri = kernel_rbf(X1.real, X2.imag, l, sigma_f)
    kir = kernel_rbf(X1.imag, X2.real, l, sigma_f)

    return krr + kii + 1j * (kri - kir)


def kernel_rbf_complex_proper(X1: np.ndarray,
                              X2: np.ndarray,
                              l: float = 1.0,
                              sigma_f: float = 1.0) -> np.ndarray:
    """
    RBF implementation for complex proper case

    Parameters
    ----------
    X1: np.ndarray
        A 2D array of `m` points (dimension is m x d).
    X2: np.ndarray
        Array of n points (n x d).
    l : float
        The length-scale of the kernel
    sigma_f : float
        The standard deviation of the kernel

    Returns
    -------
    np.ndarray
        A 2D array corresponding to the covariance function.
        Dimension is (m x n).
    """
    krr = kernel_rbf(X1.real, X2.real, l, sigma_f)
    kii = kernel_rbf(X1.imag, X2.imag, l, sigma_f)

    return krr + kii


def kernel_linear(X1: np.ndarray,
                  X2: np.ndarray,
                  bias: float = 0.0) -> np.ndarray:
    """
    Linear Kernel

    Parameters
    ----------
    X1: np.ndarray
        A 2D array of `m` points (dimension is m x d).
    X2: np.ndarray
        Array of n points (n x d).
    bias : float
        The linear kernel bias
    """
    return X1 @ X2.T.conj() + bias


def compute_loglikelihood_naive(X_train: np.ndarray,
                                Y_train: np.ndarray,
                                noise_power: float,
                                kernel: KernelFunc = kernel_rbf,
                                theta: Optional[ParamsType] = None) -> float:
    """
    Compute the loglikelihood (using a naive and less stable implementation).

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    theta : List
        Contains the kernel hyper parameters, which are passed by position to
        the kernel function after `X_train` and `Y_train`

    Returns
    -------
    float
        The log likelihood.
    """
    if theta is None:
        theta = []
    m = X_train.shape[0]
    K = kernel(X_train, X_train, *theta) + noise_power * np.eye(m)

    ll = -0.5 * np.log(np.linalg.det(K)) + \
        -0.5 * Y_train.T @ np.linalg.inv(K) @ Y_train + \
        -0.5 * m * math.log(2*np.pi)
    assert ll.size == 1
    return cast(float, ll.flatten()[0])


def compute_loglikelihood_naive_complex(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        noise_power: float,
        kernel: KernelFunc = kernel_rbf_complex_proper,
        theta: Optional[ParamsType] = None) -> float:
    """
    Compute the loglikelihood (using a naive and less stable implementation).

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    theta : List
        Contains the kernel hyper parameters, which are passed by position to
        the kernel function after `X_train` and `Y_train`

    Returns
    -------
    float
        The log likelihood.
    """
    if theta is None:
        theta = []
    m = X_train.shape[0]
    K = kernel(X_train, X_train, *theta) + noise_power * np.eye(m)

    # Note that the imaginaty part is zero
    ll = -0.5 * np.log(np.linalg.det(K)) + \
        -0.5 * Y_train.T.conj() @ np.linalg.inv(K) @ Y_train + \
        -0.5 * m * math.log(2*np.pi)
    assert ll.size == 1
    return cast(float, ll.real.flatten()[0])


def compute_loglikelihood(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        noise_power: float,
        kernel: KernelFunc = kernel_rbf,
        theta: Optional[ParamsType] = None,
        diagonal_loading: Optional[DiagLoading] = None) -> float:
    """
    Compute the loglikelihood.

    Parameters
    ----------
    X_train : np.A
        ndarray 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    theta : List
        Contains the kernel hyper parameters, which are passed by position to
        the kernel function after `X_train` and `Y_train`
    diagonal_loading : float, bool, None
        If True or equal to a floating number then a diagonal loading will be
        performed to the `K(X_train, X_train)` covariance matrix. This improve
        its conditioning number and allow taking the Cholesky decomposition
        when the matrix is close to being singular. If True is passed, then
        each diagonal element is sum with 10^-10. If a floating number if
        passed it is used as the loading.

    Returns
    -------
    float
        The log likelihood.
    """
    if theta is None:
        theta = []
    m = X_train.shape[0]
    K = kernel(X_train, X_train, *theta) + \
        noise_power * np.eye(m)

    if diagonal_loading is not None:
        if diagonal_loading is True:
            diagonal_loading = 1e-10
        K += diagonal_loading * np.eye(m)

    L = np.linalg.cholesky(K)
    ll = -np.sum(np.log(np.diagonal(L))) + \
        -0.5 * Y_train.T @ (np.linalg.lstsq(L.T, np.linalg.lstsq(L, Y_train, rcond=None)[0], rcond=None)[0]) + \
        -0.5 * m * np.log(2*np.pi)
    assert ll.size == 1
    return cast(float, ll.flatten()[0])


def compute_loglikelihood_complex(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        noise_power: float,
        kernel: KernelFunc = kernel_rbf,
        theta: Optional[ParamsType] = None,
        diagonal_loading: Optional[DiagLoading] = None) -> float:
    """
    Compute the loglikelihood.

    Parameters
    ----------
    X_train : np.A
        ndarray 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    theta : List
        Contains the kernel hyper parameters, which are passed by position to
        the kernel function after `X_train` and `Y_train`
    diagonal_loading : float, bool, None
        If True or equal to a floating number then a diagonal loading will be
        performed to the `K(X_train, X_train)` covariance matrix. This improve
        its conditioning number and allow taking the Cholesky decomposition
        when the matrix is close to being singular. If True is passed, then
        each diagonal element is sum with 10^-10. If a floating number if
        passed it is used as the loading.

    Returns
    -------
    float
        The log likelihood.
    """
    assert (Y_train.ndim == 1)
    if theta is None:
        theta = []
    m = X_train.shape[0]
    K = kernel(X_train, X_train, *theta) + \
        noise_power * np.eye(m)

    if diagonal_loading is not None:
        if diagonal_loading is True:
            diagonal_loading = 1e-10
        K += diagonal_loading * np.eye(m)

    L = np.linalg.cholesky(K)

    # Note that the imaginaty part is zero
    ll = -np.sum(np.log(np.diagonal(L))) + \
        -0.5 * Y_train.T.conj() @ (np.linalg.lstsq(L.T, np.linalg.lstsq(L, Y_train, rcond=None)[0], rcond=None)[0]) + \
        -0.5 * m * np.log(2*np.pi)
    assert ll.size == 1
    return cast(float, ll.real.flatten()[0])


def find_optimum_log_likelihood_params_real(
        X_train: np.ndarray, Y_train: np.ndarray, noise_power: float,
        kernel: KernelFunc, initial_theta: ParamsType,
        *minimizeargs: Iterable[float]) -> OptimizeResult:
    """
    Find the optimum hyperparameters `theta` for the kernel.

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    initial_theta : List
        The initial theta values.
    minimizeargs : iterable
        Extra arguments that are passed to `scipy.optimize.minimize` (See `scipy.optimize.minimize` help)

    Returns
    -------
    res : OptimizeResult
        The result of the minimization. See `scipy.optimize.minimize`.
        Particularly, the optimum values are returned by `res.x`
    """
    def nll(theta: ParamsType) -> float:
        return -1 * compute_loglikelihood_naive(X_train, Y_train, noise_power,
                                                kernel, theta)

    return minimize(nll, initial_theta, *minimizeargs)


def find_optimum_log_likelihood_params_complex(
        X_train: np.ndarray, Y_train: np.ndarray, noise_power: float,
        kernel: KernelFunc, initial_theta: ParamsType,
        *minimizeargs: Iterable[float]) -> OptimizeResult:
    """
    Find the optimum hyperparameters `theta` for the kernel.

    Parameters
    ----------
    X_train : np.ndarray
        A 2D numpy array with dimension (m x d), where `m` is the number of
        points and `d` is the number of features.
    Y_train : np.ndarray
        A 1D numpy array with size `m` containing the values corresponding to
        the `m` rows in `X_train`.
    noise_power : float
        The noise power
    kernel : function
        A kernel function that accepts `X_train` and `Y_train`. Subsequent
        arguments accepted by the kernel function are the kernel
        hyperparameters and are taken from `*theta`.
    initial_theta : List
        The initial theta values.
    minimizeargs : iterable
        Extra arguments that are passed to `scipy.optimize.minimize` (See `scipy.optimize.minimize` help)

    Returns
    -------
    res : OptimizeResult
        The result of the minimization. See `scipy.optimize.minimize`.
        Particularly, the optimum values are returned by `res.x`
    """
    def nll(theta: ParamsType) -> float:
        return -1 * compute_loglikelihood_complex(X_train, Y_train,
                                                  noise_power, kernel, theta)

    return minimize(nll, initial_theta, *minimizeargs)


# def nll_fn(X_train, Y_train, noise, naive=True, kernel=kernel_rbf):
#     """
#     Returns a function that computes the negative marginal log- likelihood for
#     training data X_train and Y_train and given noise level. Args: X_train:
#     training locations (m x d). Y_train: training targets (m x 1). noise: known
#     noise level of Y_train. naive: if True use a naive implementation of Eq.
#     (7), if False use a numerically more stable implementation. Returns:
#     Minimization objective.
#     """

#     def nll_naive(theta):
#         # Naive implementation of Eq. (7). Works well for the examples
#         # in this article but is numerically less stable compared to
#         # the implementation in nll_stable below.
#         K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
#         noise**2 * np.eye(len(X_train))
#         return 0.5 * np.log(det(K)) + \
#                0.5 * Y_train.T.dot(np.linalg.inv(K).dot(Y_train)) + \
#                0.5 * len(X_train) * np.log(2*np.pi)

#     def nll_stable(theta):
#         # Numerically more stable implementation of Eq. (7) as described
#         # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
#         # 2.2, Algorithm 2.1.
#         K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
#             noise**2 * np.eye(len(X_train))
#         L = cholesky(K)
#         return np.sum(np.log(np.diagonal(L))) + \
#            0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) + \
#            0.5 * len(X_train) * np.log(2*np.pi)

#     if naive:
#         return nll_naive
#     else:
#         return nll_stable


# Prediction Function
def posterior_predictive(
        X_s: np.ndarray,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        noise_power: float = 1e-8,
        kernel: KernelFunc = kernel_rbf,
        theta: Optional[ParamsType] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution from `m` training data
    `X_train` and `Y_train` and `n` new inputs `X_s`.

    Parameters
    ----------
    X_s:
        New input locations (n x d).
    X_train:
        Training locations (m x d).
    Y_train:
        Training targets (m x 1).
    noise_power:
        Noise parameter.
    kernel : function
        The kernel
    theta : List
        The hyperparamters of the kernel.

    Returns
    -------
    np.ndarray, np.ndarray
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    if theta is None:
        theta = []
    K = kernel(X_train, X_train, *theta) + noise_power * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, *theta)
    K_ss = kernel(X_s, X_s, *theta)
    K_inv = np.linalg.inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s
