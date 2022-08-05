import numpy as np
from scipy.integrate import quad
from sympy import true

class ClusterCovarianceModel(object):
    '''
    Setup initial cosmology and Make a ccl class. We will later vary them for MCMC
    '''
    overdensity_delta = 200
    # in km/s
    c = 299792.458 

class MassRichness(object):

    @staticmethod
    def MurataCostanzi(ln_true_mass, richness_bin, richness_bin_next, h0):
        """
        Define lognormal mass-richness relation 
        (leveraging paper from Murata et. alli - ArxIv 1707.01907 and Costanzi et al ArxIv 1810.09456v1)
       

        Args:
            ln_true_mass: ln(true mass)
            richness_bin: ith richness bin
            richness_bin_next: i+1th richness bin
            h0: 

        Returns:
            The probability that the true mass ln(ln_true_mass) is observed within 
            the bins richness_bin and richness_bin_next
        """

        alpha = 3.207           # Murata
        beta = 0.75             # Costanzi
        sigma_zero = 2.68       # Costanzi
        q = 0.54                # Costanzi
        m_pivot = 3.e+14/h0     # in solar masses , Murata and Costanzi use it

        sigma_lambda = sigma_zero+q*(ln_true_mass - np.log(m_pivot))
        average = alpha+beta*(ln_true_mass-np.log(m_pivot))

        def integrand(richness):
            return (1./richness) * np.exp(-(np.log(richness)-average)**2. / (2.*sigma_lambda**2.))\
                / (np.sqrt(2.*np.pi) * sigma_lambda)
        
        return (quad(integrand, richness_bin, richness_bin_next)[0])