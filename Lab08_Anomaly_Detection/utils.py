import numpy as np
import matplotlib.pyplot as plt


def multivariate_gaussian(X, mu, var):
    k = len(mu)

    if var.ndim == 1:
        var = np.diag(var)

        X = X - mu
        p = (2 * np.pi) ** (-k / 2) * np.linalg.det(var) ** (-0.5) * \
            np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

        return p


def visualize_fit(X, mu, var):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)
    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10 ** (np.arange(-20., 1, 3)), linewidths=1)

    plt.title('The Gaussian contours of the distribution fit to the dataset')
    plt.ylabel('Throughput (mb/s)')
    plt.xlabel('Latency (ms)')