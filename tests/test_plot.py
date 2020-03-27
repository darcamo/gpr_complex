# import math

# import numpy as np
# from bokeh.plotting import output_file
# # from sklearn.gaussian_process import GaussianProcessRegressor
# # from sklearn.gaussian_process.kernels import RBF as RBF_sklearn
# # from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

# from gpr_complex.kernels import RBF
# from gpr_complex.model import GPR
# from gpr_complex.plot import plot_gp

# output_file("lalala.html")

# noise_power = 0.5**2

# def compute(x: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring
#     return 2.5 * x + math.sqrt(noise_power) * np.random.randn(*(x.shape))

# x: np.ndarray = np.random.randn(200, 1)
# y: np.ndarray = compute(x)

# kernel = RBF(l=1.0, sigma_f=1.0)
# gp = GPR(noise_power=noise_power, kernel=kernel)
# gp.fit(x, y)

# # Sklearn GPR
# # kernel2 = ConstantKernel() * RBF_sklearn() + WhiteKernel()
# # gp2 = GaussianProcessRegressor(kernel=kernel2)
# # gp2.fit(x, y)

# x_test = np.linspace(-1.5, 1.5, 30)[:, np.newaxis]
# y_test = compute(x_test)

# y_pred, cov_pred = gp.predict(x_test, return_cov=True)
# # y_pred2, cov_pred2 = gp2.predict(x_test, return_cov=True)

# # x_test = x_test.flatten()
# # y_pred = y_pred.flatten()

# indexes = np.argsort(x_test.flatten())

# plot_gp(dict(x=x_test.flatten()[indexes],
#              mu=y_pred.flatten()[indexes],
#              cov=cov_pred),
#         dict(x=x.ravel(), y=y.ravel()),
#         samples=3)
