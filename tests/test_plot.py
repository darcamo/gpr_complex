import numpy as np

from bokeh.plotting import output_file
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from gpr_complex.plot import plot_gp

output_file("lalala.html")


def compute(x: np.ndarray) -> np.ndarray:  # pylint: disable=missing-function-docstring
    return 2.5 * x + 0.5 * np.random.randn(*(x.shape))


x: np.ndarray = np.random.randn(200, 1)
y: np.ndarray = compute(x)

kernel = ConstantKernel() * RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(x, y)

x_test = np.linspace(-1.5, 1.5, 30)[:, np.newaxis]
y_test = compute(x_test)

y_pred, cov_pred = gp.predict(x_test, return_cov=True)

x_test = x_test.flatten()
y_pred = y_pred.flatten()
indexes = np.argsort(x_test)

plot_gp(dict(x=x_test[indexes], mu=y_pred[indexes], cov=cov_pred),
        dict(x=x.ravel(), y=y.ravel()),
        samples=3)
