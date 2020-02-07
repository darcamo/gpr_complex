import math
import numpy as np
from gpr_complex.model import GPR
from gpr_complex.kernels import RBF

num_features = 4
num_samples = 200
noise_power = 1e-5
rbf = RBF(1.0, 1.0)

X1 = np.random.randn(num_samples, num_features)
y1 = X1 @ [1, -2, 3, -4] + \
    math.sqrt(noise_power) * np.random.randn(num_samples)

gpr = GPR(noise_power, rbf)
print(gpr.kernel.get_params())
gpr.fit(X1, y1)
print(gpr.kernel.get_params())

y1_pred, y1_pred_cov = gpr.predict(X1, return_cov=True)
print("y1", y1.shape)
print("y1_pred", y1_pred.shape)
print("Mean Squared Error", np.mean((y1 - y1_pred)**2))
