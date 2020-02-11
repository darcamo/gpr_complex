import unittest
import numpy as np
from itertools import product
import math

from gpr_complex.gaussian_process import kernel_rbf_complex_proper, kernel_rbf, kernel_rbf_complex, kernel_linear, compute_loglikelihood_naive_complex, compute_loglikelihood_naive, compute_loglikelihood_complex, compute_loglikelihood


class TestKernels(unittest.TestCase):  # pylint: disable=missing-class-docstring
    def setUp(self) -> None:
        m = 5  # Number of points
        d = 2  # Number of features
        self.m = m
        self.d = d
        self.X1 = np.random.randn(m, d)
        self.X2 = np.random.randn(m, d)

    def test_kernel_rbf(self) -> None:  # pylint: disable=missing-function-docstring
        m, X1, X2 = self.m, self.X1, self.X2

        # Hyper parameters
        l = 2.3
        sigma_f = 1.5

        cov = kernel_rbf(X1, X2, l=l, sigma_f=sigma_f)
        self.assertEqual(cov.shape, (m, m))

        expected_cov = np.empty(shape=(m, m))
        for i, j in product(range(m), range(m)):
            sqrdist = np.sum((X1[i] - X2[j])**2)
            expected_cov[i, j] = sigma_f**2 * math.exp(-0.5 * sqrdist / l**2)

        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_kernel_rbf_complex(self) -> None:  # pylint: disable=missing-function-docstring
        m, d = self.m, self.d
        X1 = np.random.randn(m, d) + 1j * np.random.randn(m, d)
        X2 = np.random.randn(m, d) + 1j * np.random.randn(m, d)

        # Hyper parameters
        l = 2.3
        sigma_f = 1.5

        cov = kernel_rbf_complex(X1, X2, l=l, sigma_f=sigma_f)
        self.assertEqual(cov.shape, (m, m))

        expected_cov = np.empty(shape=(m, m), dtype=complex)
        for i, j in product(range(m), range(m)):
            krr = kernel_rbf(X1.real[i:i + 1],
                             X2.real[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)
            kii = kernel_rbf(X1.imag[i:i + 1],
                             X2.imag[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)
            kri = kernel_rbf(X1.real[i:i + 1],
                             X2.imag[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)
            kir = kernel_rbf(X1.imag[i:i + 1],
                             X2.real[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)
            # sqrdist = np.linalg.norm(d)**2
            expected_cov[i, j] = krr + kii + 1j * (kri - kir)
        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_kernel_rbf_complex_proper(self) -> None:  # pylint: disable=missing-function-docstring
        m, d = self.m, self.d
        X1 = np.random.randn(m, d) + 1j * np.random.randn(m, d)
        X2 = np.random.randn(m, d) + 1j * np.random.randn(m, d)

        # Hyper parameters
        l = 2.3
        sigma_f = 1.5

        cov = kernel_rbf_complex_proper(X1, X2, l=l, sigma_f=sigma_f)
        self.assertEqual(cov.shape, (m, m))

        expected_cov = np.empty(shape=(m, m), dtype=float)
        for i, j in product(range(m), range(m)):
            krr = kernel_rbf(X1.real[i:i + 1],
                             X2.real[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)
            kii = kernel_rbf(X1.imag[i:i + 1],
                             X2.imag[j:j + 1],
                             l=l,
                             sigma_f=sigma_f)

            # sqrdist = np.linalg.norm(d)**2
            expected_cov[i, j] = krr + kii
        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_kernel_linear(self) -> None:  # pylint: disable=missing-function-docstring
        m, X1, X2 = self.m, self.X1, self.X2

        # hyper parameter
        bias = 1.0

        cov = kernel_linear(X1, X2, bias=bias)
        self.assertEqual(cov.shape, (m, m))

        expected_cov = np.empty(shape=(m, m))
        for i, j in product(range(m), range(m)):
            expected_cov[i, j] = X1[i] @ X2[j].T + bias

        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_kernel_linear_complex(self) -> None:  # pylint: disable=missing-function-docstring
        m, d = self.m, self.d
        X1 = np.random.randn(m, d) + 1j * np.random.randn(m, d)
        X2 = np.random.randn(m, d) + 1j * np.random.randn(m, d)

        # hyper parameter
        bias = 1.0

        cov = kernel_linear(X1, X2, bias=bias)
        self.assertEqual(cov.shape, (m, m))

        expected_cov = np.empty(shape=(m, m), dtype=complex)
        for i, j in product(range(m), range(m)):
            expected_cov[i, j] = X1[i] @ X2[j].T.conj() + bias

        np.testing.assert_array_almost_equal(cov, expected_cov)

    def test_compute_loglikelihood_naive(self) -> None:  # pylint: disable=missing-function-docstring
        m = self.m
        X1 = self.X1
        y1 = X1 @ [1, -2]
        noise_power = 0.0
        l = 1.0
        sigma_f = 1.0
        nll = -compute_loglikelihood_naive(
            X1, y1, noise_power, kernel=kernel_rbf, theta=[l, sigma_f])

        # Compute the expected likelihood
        K = kernel_rbf(X1, X1, l=l, sigma_f=sigma_f)
        expected_nll = 0.5 * np.log(np.linalg.det(K)) + \
            +0.5 * y1.T @ np.linalg.inv(K).dot(y1) + \
            +0.5 * m * math.log(2*np.pi)
        self.assertAlmostEqual(nll, expected_nll)

    def test_compute_loglikelihood(self) -> None:  # pylint: disable=missing-function-docstring
        m = self.m
        X1 = self.X1
        y1 = X1 @ [1, -2]
        noise_power = 0.0
        l = 1.0
        sigma_f = 1.0
        nll = -compute_loglikelihood(
            X1, y1, noise_power, kernel=kernel_rbf, theta=[l, sigma_f])

        # Compute the expected likelihood
        K = kernel_rbf(X1, X1, l=l, sigma_f=sigma_f)
        expected_nll = 0.5 * np.log(np.linalg.det(K)) + \
            +0.5 * y1.T @ np.linalg.inv(K).dot(y1) + \
            +0.5 * m * math.log(2*np.pi)
        self.assertAlmostEqual(nll, expected_nll)

    def test_compute_loglikelihood_naive_complex(self) -> None:  # pylint: disable=missing-function-docstring
        m, d = self.m, self.d
        X1 = np.random.randn(m, d) + 1j * np.random.randn(m, d)
        y1 = X1 @ [1, -2]
        noise_power = 0.0
        l = 1.0
        sigma_f = 1.0
        nll = -compute_loglikelihood_naive_complex(
            X1,
            y1,
            noise_power,
            kernel=kernel_rbf_complex_proper,
            theta=[l, sigma_f])

        # Compute the expected likelihood
        K = kernel_rbf_complex_proper(X1, X1, l=l, sigma_f=sigma_f)
        expected_nll = np.abs(0.5 * np.log(np.linalg.det(K)) +
                              +0.5 * y1.T.conj() @ np.linalg.inv(K).dot(y1) +
                              +0.5 * m * math.log(2 * np.pi))
        self.assertAlmostEqual(nll, expected_nll)

    def test_compute_loglikelihood_complex(self) -> None:  # pylint: disable=missing-function-docstring
        m, d = self.m, self.d
        X1 = np.random.randn(m, d) + 1j * np.random.randn(m, d)
        y1 = X1 @ [1, -2]
        noise_power = 0.0
        l = 1.0
        sigma_f = 1.0
        nll = -compute_loglikelihood_complex(X1,
                                             y1,
                                             noise_power,
                                             kernel=kernel_rbf_complex_proper,
                                             theta=[l, sigma_f])

        # Compute the expected likelihood
        K = kernel_rbf_complex_proper(X1, X1, l=l, sigma_f=sigma_f)
        expected_nll = np.abs(0.5 * np.log(np.linalg.det(K)) +
                              +0.5 * y1.T.conj() @ np.linalg.inv(K).dot(y1) +
                              +0.5 * m * math.log(2 * np.pi))
        self.assertAlmostEqual(nll, expected_nll)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    np.set_printoptions(precision=2, linewidth=120)
    unittest.main()
