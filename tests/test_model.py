import unittest
import numpy as np
import math
from gpr_complex.kernels import RBF_ComplexProper
from gpr_complex.model import GPR


def f(x: np.ndarray, noise_power: float) -> np.ndarray:
    assert x.shape[1] == 1
    num_samples = x.shape[0]
    return 3 * x[:, 0] + math.sqrt(noise_power) * (
        np.random.randn(num_samples) + 1j * np.random.randn(num_samples))


class TestGPR(unittest.TestCase):
    def setUp(self) -> None:
        num_samples = 50
        noise_power = 1e-2
        self.x = np.random.randn(num_samples,
                                 1) + 1j * np.random.randn(num_samples, 1)
        self.y = f(self.x, noise_power)

        self.x_test = np.random.randn(
            num_samples, 1) + 1j * np.random.randn(num_samples, 1)
        self.y_test = f(self.x_test, noise_power)

        # kernel = RBF_ComplexProper(1.0, 1.0)

    def test_methods_and_properties(self) -> None:
        noise_power = 1e-2
        kernel = RBF_ComplexProper(1.0, 1.0)
        gpr = GPR(noise_power, kernel)

        self.assertEqual(gpr.input_dim, 0)
        self.assertFalse(gpr.is_trained)
        self.assertIsNone(gpr.likelihood)
        self.assertEqual(gpr.kernel.get_params(), [1.0, 1.0])

        gpr.fit(self.x, self.y)

        self.assertEqual(gpr.input_dim, 1)
        self.assertTrue(gpr.is_trained)
        self.assertIsNotNone(gpr.likelihood)
        params = gpr.kernel.get_params()
        self.assertNotAlmostEqual(params[0], 1.0)
        self.assertNotAlmostEqual(params[1], 1.0)

        np.testing.assert_array_almost_equal(self.x, gpr._x_train)
        np.testing.assert_array_almost_equal(self.y, gpr._y_train)

        # Add more data without re-training the model
        num_samples = 20
        noise_power = 0.01
        x = np.random.randn(num_samples,
                            1) + 1j * np.random.randn(num_samples, 1)
        y = f(x, noise_power)

        # Perform prediction
        y_pred1 = gpr.predict(x)
        mse1 = np.mean(np.abs(y_pred1 - y)**2)

        # Add new data to the model
        gpr.add_new_data(x, y)

        # Test that the kernel parameters didn't change with the new data
        params2 = gpr.kernel.get_params()
        np.testing.assert_array_almost_equal(params, params2)

        # Test that the new data was added
        assert (gpr._x_train is not None)
        assert (gpr._y_train is not None)
        self.assertEqual(gpr._x_train.shape[0], 70)
        self.assertEqual(gpr._y_train.size, 70)

        # Reset the model by fitting it again
        gpr.fit(self.x, self.y)

        # Now call the predict_and_add_new_data method
        y_pred2 = gpr.predict_and_add_new_data(x, y)
        mse2 = np.mean(np.abs(y_pred2 - y)**2)
        self.assertLess(mse2, mse1)


if __name__ == '__main__':
    unittest.main()

# import math
# import numpy as np
# from gpr_complex.model import GPR
# from gpr_complex.kernels import RBF

# num_features = 4
# num_samples = 200
# noise_power = 1e-5
# rbf = RBF(1.0, 1.0)

# X1 = np.random.randn(num_samples, num_features)
# y1 = X1 @ [1, -2, 3, -4] + \
#     math.sqrt(noise_power) * np.random.randn(num_samples)

# gpr = GPR(noise_power, rbf)
# print(gpr.kernel.get_params())
# gpr.fit(X1, y1)
# print(gpr.kernel.get_params())

# y1_pred, y1_pred_cov = gpr.predict(X1, return_cov=True)
# print("y1", y1.shape)
# print("y1_pred", y1_pred.shape)
# print("Mean Squared Error", np.mean((y1 - y1_pred)**2))
