import unittest

if __name__ == '__main__':
    # unittest.main()

    import math
    import numpy as np
    from gpr_complex.kernels import RBF

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
