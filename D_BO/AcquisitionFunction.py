import math
import numpy as np
import scipy
from scipy.stats import norm


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, acq, inSpaces):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.acq = acq
        self.spaces = inSpaces
        self.acq_name = self.acq['name']
        self.idxDiscrete = self.acq['discrete_idx']
        self.idxCat = self.acq['cat_idx']
        self.stillGood = True
        self.isSame = False
        self.decrement = 2
        self.currentEpsilon = 0.01

        self.dens_good = None
        self.dens_bad = None
        self.varType = ""
        for i in range(0,len(self.idxCat)):
            self.varType += 'u'

    def setEpsilon(self, val):
        self.acq['epsilon'] = val

    def setBetaT(self, val):
        self.acq['betaT'] = val

    def setIte(self, val):
        self.ite = val

    def setDim(self, val):
        self.dim = val

    def setEsigma(self, val):
        self.esigma = val

    @staticmethod
    def resampleFromKDE(kde, size):
        n, d = kde.data.shape
        indices = np.random.randint(0, n, size)

        print("indices",indices)
        cov = np.diag(kde.bw) ** 2
        means = kde.data[indices, :]
        norm = np.random.multivariate_normal(np.zeros(d), cov, size)

        print("norm",norm)
        return np.transpose(means + norm)

    def getEpsilon(self):
        return self.acq['epsilon']

    def acq_kind(self, x, gp, y_max, **args):
        if np.any(np.isnan(x)):
            return 0
        if self.acq_name == 'ucb':
            return self._ucb(x, gp, self.acq['epsilon'])
        if self.acq_name == 'ucb_opteta':
            return self._ucb_opteta(x, gp, self.dim, self.ite, self.acq['epsilon'], self.acq['betaT'], self.idxDiscrete, self.idxCat, self.spaces, args.get("obs"), self.esigma)

    @staticmethod
    def _pi(x, gp, fMax, epsilon):
        mean, _, var = gp.predictScalar(x)
        #var[var < 1e-10] = 0
        std = np.sqrt(var)
        Z = (mean - fMax - epsilon) / std
        result = np.matrix(scipy.stats.norm.cdf(Z))
        return result

    @staticmethod
    def _ei(x, gp, fMax, discrete_idx, epsilon, stillGood, currentEps):

        epsilon = 0.01

        mean, _, var = gp.predictScalarLib(x)

        var2 = np.maximum(var, 1e-4 + 0 * var)
        #var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)

        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result

    @staticmethod
    def _eiT(x, gp, fMax, discrete_idx, epsilon, stillGood, currentEps):
        epsilon = 0.01
        mean, _, var = gp.predictScalarTrans(x)
        # mean, _, var = gp.predictCholeskyScalar(x)
        var2 = np.maximum(var, 1e-16 + 0 * var)
        # var[var < 1e-10] = 0
        std = np.sqrt(var2)
        Z = (mean - fMax - epsilon) / (std)
        result = (mean - fMax - epsilon) * scipy.stats.norm.cdf(Z) + std * scipy.stats.norm.pdf(Z)
        return result

    @staticmethod
    def _ucb(x, gp, beta):

        mean, _, var = gp.predictScalarLib(x)

        std = np.sqrt(var)
        result = np.matrix(mean + beta * std)
        return result

    @staticmethod



    def _ucb_opteta(x, gp, dim, ite, inEps, betaT, discrete_idx, cat_idx, spaces, obs, sigma_dup=0.01):


        mean, _, var = gp.predictScalarLib(x)
        std = np.sqrt(var)


        inBeta = inEps


        result = (mean - inBeta * std) * 1 * 1# + penalty

        return result
