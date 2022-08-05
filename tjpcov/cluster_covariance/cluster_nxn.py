
import numpy as np
import pyccl as ccl
# this will be necessary for some integrals
from scipy.integrate import romb, quad
from tjpcov.cluster_covariance import ClusterCovariance, ClusterCovarianceArgs
import time
from collections import namedtuple

class ClusterNxN(ClusterCovariance):

    def __init__(self, args : ClusterCovarianceArgs, cosmo):
        
        super().__init__(args, cosmo)   

    def calculate_covariance_all(self, verbose=True, romberg_num = 2**6+1):

        ini = time.time()

        # TODO Improve workflow
        self.eval_true_vec(romberg_num)
        self.eval_M1_true_vec(romberg_num)

        final_array = np.zeros((self.num_richness_bins, self.num_richness_bins, self.num_z_bins, self.num_z_bins))

        if verbose:
            print("Zbin_i \tZbin_j \tLbd_i \tLbd_j \tCov \t\tDtime")

        matrixElement = namedtuple('matrixElement', ['li', 'lj', 'zi', 'zj'])
        elemList = []
        for richness_i in range(self.num_richness_bins):   
            for richness_j in range(richness_i, self.num_richness_bins):
                for z_i in range (self.num_z_bins):
                    for z_j in range(z_i, self.num_z_bins):
                        elemList.append(matrixElement(richness_i, richness_j, z_i, z_j))

        for e in elemList:
            start = time.time()
            shot_plus_cov = self._covariance_step(e.zi, e.zj, e.li, e.lj, romberg_num)
            cov_term = shot_plus_cov
            final_array[e.li,e.lj,e.zi,e.zj]= cov_term
            final_array[e.li,e.lj,e.zj,e.zi]= final_array[e.li,e.lj,e.zi,e.zj]
            final_array[e.lj,e.li,e.zi,e.zj]= final_array[e.li,e.lj,e.zi,e.zj]      
            end = time.time()
            if verbose:
                print ( "%d \t%d \t%d \t%d \t%4.3e \t%3.2f "%\
                        (z_i, z_j, richness_i, richness_j,float(final_array[e.li,e.lj,e.zi,e.zj]),(end-start)))

        print ("\ntotal elapsed matrix for matrix was: " + str((end-ini)/3600) + "hours.")

        return final_array

    def _covariance_step(self, z_i,z_j,richness_i,richness_j, romberg_num):

        if (richness_i == richness_j and z_i == z_j):
            shot = self.shot_noise(z_i,richness_i)
        else:
            shot = 0

        cov = self._covariance_NxN(z_i,z_j,richness_i,richness_j, romberg_num)
        
        return shot + cov

    def _covariance_NxN(self, bin_z_i, bin_z_j, bin_richness_i, bin_richness_j, romberg_num):
        """ Cluster counts covariance
        Args:
            bin_z_i (float or ?array): tomographic bins in z_i or z_j
            bin_lbd_i (float or ?array): bins of richness (usually log spaced)

        Returns:
            float: Covariance at given bins

        """
        dz = (self.Z1_true_vec[bin_z_i, -1]-self.Z1_true_vec[bin_z_i, 0])/(romberg_num-1)
        
        partial_vec = [
            self.partial2(self.Z1_true_vec[bin_z_i, m], bin_z_j, bin_richness_j) for m in range(romberg_num)
        ]
        
        romb_vec = partial_vec*self.dV_true_vec[bin_z_i]*self.M1_true_vec[bin_richness_i, bin_z_i]*self.G1_true_vec[bin_z_i]

        return (self.survey_area**2)*romb(romb_vec, dx=dz)


    def _get_min_radial_idx(self):
        return np.argwhere(self.r_vec  < 0.95*self.radial_lower_limit)[-1][0]
    
    def _get_max_radial_idx(self):
        return np.argwhere(self.r_vec  > 1.05*self.radial_upper_limit)[0][0]


    def eval_true_vec(self, romb_num):
        """ Pre computes the -geometric- true vectors  
        Z1, G1, dV for Cov_N_N. 

        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 

        Returns:
            (array) Z1_true_vec
            (array) G1_true_vec
            (array) dV_true_vec

        """

        cosmo = self.cosmo
        Num_z_bins = self.num_z_bins
        Lim_z_min, Lim_z_max = self.z_lower_limit, self.z_upper_limit   
        Z_bins = self.z_bins
        Z_bin_range = self.z_bin_range

        self.Z1_true_vec = np.zeros((Num_z_bins, romb_num))
        self.G1_true_vec = np.zeros((Num_z_bins, romb_num))
        self.dV_true_vec = np.zeros((Num_z_bins, romb_num))

        for i in range(Num_z_bins):
            self.Z1_true_vec[i] = np.linspace(
                max(Lim_z_min, Z_bins[i]-4*Z_bin_range),
                min(Lim_z_max, Z_bins[i+1]+6*Z_bin_range), romb_num)
            self.G1_true_vec[i] = ccl.growth_factor(cosmo, 1/(1+self.Z1_true_vec[i]))
            self.dV_true_vec[i] = [ self.dV(self.Z1_true_vec[i, m], i)
                              for m in range(romb_num)]


    def eval_M1_true_vec(self, romb_num):
        """ Pre computes the true vectors  
        M1 for Cov_N_N. 

        Args:
            (int) romb_num: controls romb integral precision. 
                        Typically 10**6 + 1 
        """

        print('evaluating M1_true_vec (this may take some time)...')
        Num_Lbd_bins = self.num_richness_bins
        Num_z_bins = self.num_z_bins

        self.M1_true_vec = np.zeros((Num_Lbd_bins, Num_z_bins, romb_num))
        for lbd in range(Num_Lbd_bins):
            print(f'\r{100*(lbd)/Num_Lbd_bins :.1f} %', end="")

            for z in range(Num_z_bins):

                for m in range(romb_num):
                    self.M1_true_vec[lbd, z, m] = \
                        self.integral_mass( self.Z1_true_vec[z, m], lbd)
        print(f'\r100.')

    def eval_sigma_vec(self):
        sigma_vec = np.zeros((self.nz,self.nz))

        for i in range (self.nz):
            for j in range (i, self.nz):

                sigma_vec[i,j] = self.double_bessel_integral(self.z_true_vec[i],self.z_true_vec[j])
                sigma_vec[j,i] = sigma_vec[i,j]

        return sigma_vec