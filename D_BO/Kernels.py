import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.gaussian_process.kernels import Matern as skMatern


class Kernels:
    def __init__(self,inputName):
        self.name = inputName
        self.lengthScale = 1.0
        self.amp = 1.0

    def k(self, x, xprime):
        if self.name == 'matern':
            return self._mattern_kernel(x,xprime,self.lengthScale, self.amp)
        if self.name == 'ise':
            return self._ise_kernel(x,xprime,self.lengthScale, self.amp)
        if self.name == 'rbs':
            return self._radial_basis_kernel(x,xprime,self.lengthScale,self.amp)
        if self.name == 'mixed':
            return self._mixed_basis_kernel(x,xprime,self.lengthScale,self.amp)

    @staticmethod
    def _mattern_kernel(x, xprime, lengthScale, amp):
        x = np.array(x)
        xprime = np.array(xprime)
        K = skMatern(nu=amp, length_scale=lengthScale)
        return K(x, xprime)

    @staticmethod
    def _ise_kernel(x, xprime, lengthScale, amp):
        ''' Isotropic squared exponential kernel. Computes a covariance matrix from points in X1 and X2.
        Args:   X1: Array of m points (m x d).
                X2: Array of n points (n x d).
        Returns: Covariance matrix (m x n). '''
        x = np.array(x)
        xprime = np.array(xprime)
        sqdist = np.sum(x ** 2, 1).reshape(-1, 1) + np.sum(xprime ** 2, 1) - 2 * np.dot(x, xprime.T)
        K = amp * np.exp(-0.5 / (lengthScale ** 2) * sqdist)
        # print("squdist: ",sqdist.shape)
        return K # Integrated squared error

    @staticmethod
    def _radial_basis_kernel(x,xprime,len,amp):

        result = amp * np.exp(-0.5 * euclidean_distances(x, xprime) / len)
        return result

    @staticmethod
    def _mixed_basis_kernel(x, xprime, lengthScale, amp):

        x = np.array(x)
        xprime = np.array(xprime)
        p = 3
        sqdist = euclidean_distances(x, xprime)
        K = amp * np.exp(-2 * np.sin(np.pi * sqdist / p) ** 2 / (lengthScale ** 2) )
        return K
