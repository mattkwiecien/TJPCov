import time
from abc import ABC, abstractmethod

import numpy as np
import pyccl as ccl
import pyccl.halos.hmfunc as hmf
# this will be necessary for some integrals
from scipy.integrate import quad, romb
from scipy.special import gamma 
from scipy.interpolate import interp1d

from tjpcov.cluster_covariance import MassRichness, ClusterCovarianceArgs, ClusterCovarianceModel


class ClusterCovariance(ABC):
    """ Covariance of Cluster + 3x2pt
    version 1 , date: - ?? ?? 2019

    Covariance matrix for a LSST-like case, using CCL packages

    Evaluate here:

    - NxN & NxCls (gg, gk, kk) 
    - Assuming full sky for now
    - Included shot noise (1-halo term)
    - Added full matrix in the end 

    TODO: verify the other limber auxiliary functions and new
    shot noise
    """
    c = ClusterCovarianceModel.c  # km/s
    bias_fft = 1.4165
    ko = 1e-4
    kmax = 3 # TODO check if this is 3 or 4
    N = 1024

    # Lazy loaded properties
    _z_true_vec = None
    @property 
    def z_true_vec(self):
        if self._z_true_vec is None:
            # Incorrect for ClxCl - maybe defer at this point to concrete method and use Lim_Z_min etc for cl x cl
            self._z_true_vec = np.linspace(self.z_bins[0], self.z_bins[self.num_z_bins], self.nz)
        return self._z_true_vec

    _z_true_vec_inv = None
    @property 
    def z_true_vec_inverse(self):
        if self._z_true_vec_inv is None:
            self._z_true_vec_inv = self.z_true_vec[::-1]
        return self._z_true_vec_inv

    _sigma_vec = None
    @property
    def sigma_vec(self):
        if self._sigma_vec is None:
            self._sigma_vec = self.eval_sigma_vec(self.nz)
        return self._sigma_vec

    def __init__(self, args: ClusterCovarianceArgs, cosmo, ovdelta=200, survey_area=4*np.pi):

        self.mass_func = hmf.MassFuncTinker10(cosmo)
        self.nz = args.nz

        # Setup Richness Bins
        self.min_richness = args.min_richness
        self.max_richness = args.max_richness
        self.richness_bin_range = args.richness_bin_range
        self.num_richness_bins = round((self.max_richness-self.min_richness)/self.richness_bin_range)
        self.richness_bins = np.round(np.logspace(np.log10(self.min_richness),
                                             np.log10(self.max_richness), 
                                             self.num_richness_bins+1), 2)

        # Define arrays for bins for Photometric z and z grid
        self.z_bin_range = args.z_bin_range
        self.num_z_bins = round((args.z_max-args.z_min)/self.z_bin_range)
        self.z_bins = np.round(np.linspace(args.z_min, args.z_max, self.num_z_bins+1), 2)

        #   minimum log mass in solar masses; 
        self.min_mass = args.min_mass
        #   maximum log mass in solar masses; above this HMF < 10^-10
        self.max_mass = args.max_mass

        # Cosmology
        self.survey_area = survey_area
        self.cosmo = cosmo
        self.ovdelta = ovdelta
        self.h0 = self.cosmo.h

        # FFT parameters:
        ko = self.ko
        kmax = self.kmax
        self.ro = 1/kmax
        self.rmax = 1/ko
        self.G = np.log(kmax/ko)
        self.L = 2*np.pi*self.N/self.G
        self.k_vec = np.logspace(np.log(self.ko), np.log(self.kmax), self.N, base=np.exp(1))
        self.r_vec = np.logspace(np.log(self.ro), np.log(self.rmax), self.N, base=np.exp(1))

        #minimum z_true for the integrals. I am assuming z_true>0.02 
        self.z_lower_limit = max(0.02, self.z_bins[0]-4*self.z_bin_range) 
        #maximum z_true for the integrals, assuming 40% larger than max z,
        # so we dont need to go till infinity
        self.z_upper_limit = self.z_bins[-1]+6*self.z_bin_range 
        
        # TODO: optimize these ranges like Nelson did
        # interpolation limits for double_bessel_integral
        # zmin & zmax drawn from Z_true_vec
        self.radial_lower_limit = self.radial_distance(self.z_lower_limit)
        self.radial_upper_limit = self.radial_distance(self.z_upper_limit)
        self.imin = self._get_min_radial_idx()
        self.imax = self._get_max_radial_idx()       

        # Do it ONCE
        self.pk_vec = ccl.linear_matter_power(self.cosmo, self.k_vec, 1)
        self.fk_vec = (self.k_vec/self.ko)**(3. - self.bias_fft)*self.pk_vec
        self.Phi_vec = np.conjugate(np.fft.rfft(self.fk_vec))/self.L

    def radial_distance(self,z):
        return ccl.comoving_radial_distance(self.cosmo, 1/(1+ z))

    def printer(self):
        print(self.N, self.ko)

    def photoz(self, z_true, z_i, sigma_0=0.05):
        """ 
        Evaluation of Photometric redshift (Photo-z),given true redshift 
        z_true and photometric bin z_i
        
        Note: Z_bins & z_i is a bad choice of variables!
                check how one use the photoz function and adapt it !

        Note: I am truncating the pdf, so as to absorb the negative redshifts 
            into the positive part of the pdf

        Args:
            z_true ()
            z_i
            sigma_0 (float): set as 0.05

        Returns: 
            (array)
        """

        sigma_z = sigma_0*(1+z_true)

        def integrand(z_phot): 
            return np.exp(- (z_phot - z_true)**2.\
                                            / (2.*sigma_z**2.)) \
                                            / (np.sqrt(2.*np.pi) * sigma_z)
        
        integral = (quad(integrand, self.z_bins[z_i], self.z_bins[z_i+1])[0]
                    / (1.-quad(integrand, -np.inf, 0.)[0]))

        return integral


    def dV(self, z_true, z_i):
        ''' Evaluates the comoving volume per steridian as function of 
        z_true for a photometric redshift bin in units of Mpc**3

        Returns:
            dv(z) = dz*dr/dz(z)*(r(z)**2)*photoz(z, bin z_i)
        '''

        cosmo = self.cosmo
        h0 = self.h0

        dV = self.c*(ccl.comoving_radial_distance(cosmo, 1/(1+z_true))**2)\
            / (100*h0*ccl.h_over_h0(cosmo, 1/(1+z_true)))\
            * (self.photoz(z_true, z_i))
        return dV


    def mass_richness(self, ln_true_mass, lbd_i):
        """ returns the probability that the true mass ln(M_true) is observed within 
        the bins lambda_i and lambda_i + 1
        """
        richness_bin = self.richness_bins[lbd_i]
        richness_bin_next = self.richness_bins[lbd_i+1]
        return MassRichness.MurataCostanzi(ln_true_mass, richness_bin, richness_bin_next, self.h0)

    def integral_mass(self, z, lbd_i):
        """ z is the redshift; i is the lambda bin,with lambda from 
        Lambda_bins[i] to Lambda_bins[i+1]

        note: ccl.function returns dn/dlog10m, I am changing integrand below 
        to d(lnM)
        """

        # TODO Replace halo_bias with HaloBias
        f = lambda ln_m: (1/np.log(10.))\
            * self.mass_func.get_mass_function(self.cosmo, np.exp(ln_m), 1/(1+z))\
            * ccl.halo_bias(self.cosmo, np.exp(ln_m), 1/(1+z),
                            overdensity=ClusterCovarianceModel.overdensity_delta)\
            * self.mass_richness( ln_m, lbd_i)

        return quad(f, self.min_mass, self.max_mass)[0]


    def integral_mass_no_bias (self, z,lbd_i):
        """ Integral mass for shot noise function
        """
        f = lambda ln_m:(1/np.log(10))*\
            self.mass_func.get_mass_function(self.cosmo, np.exp(ln_m), 1/(1+z))\
                *self.mass_richness(ln_m, lbd_i)
        # Remember ccl.function returns dn/dlog10m, I am changing 
        #   integrand to d(lnM)
        return quad(f, self.min_mass, self.max_mass)[0]


    def Limber(self, z):
        """ Calculating Limber approximation for double Bessel 
        integral for l equal zero
        """

        return ccl.linear_matter_power(self.cosmo,
                                       0.5/ccl.comoving_radial_distance(
                                        self.cosmo, 1/(1+z)), 1)/(4*np.pi)


    def cov_Limber(self, z_i, z_j, lbd_i, lbd_j):
        """Calculating the covariance of diagonal terms using Limber (the delta 
        transforms the double redshift integral into a single redshift integral)

        CAUTION: hard-wired ovdelta and survey_area!

        """
        cosmo = self.cosmo
        ovdelta = self.ovdelta
        survey_area = self.survey_area

        def integrand(z_true): 
            return self.dV(cosmo, z_true, z_i)\
            * (ccl.growth_factor(cosmo, 1/(1+z_true))**2)\
            * self.photoz(z_true, z_j)\
            * self.integral_mass(cosmo, z_true, lbd_i, self.min_mass, self.max_mass, ovdelta)\
            * self.integral_mass(cosmo, z_true, lbd_j, self.min_mass, self.max_mass, ovdelta)\
            * self.Limber(cosmo, z_true)
        return (survey_area**2)*quad(integrand, self.z_lower_limit, self.z_upper_limit)[0]


    def shot_noise(self, z_i, lbd_i):
        """Evaluates the Shot Noise term
        """
        cosmo = self.cosmo
        survey_area = self.survey_area

        h0 = self.h0
        survey_area = self.survey_area

        def integrand(z): 
            return self.c*(ccl.comoving_radial_distance(cosmo,
                                                                 1/(1+z))**2)\
            / (100*h0*ccl.h_over_h0(cosmo, 1/(1+z)))\
            * self.integral_mass_no_bias(z, lbd_i)\
            * self.photoz(z, z_i)  # TODO remove the bias!

        result = quad(integrand, self.z_lower_limit, self.z_upper_limit)
        return survey_area*result[0]


    def I_ell(self, m, R):
        """Calculating the function M_0_0
        the formula below only valid for R <=1, l = 0,  
        formula B2 ASZ and 31 from 2-fast paper 

        """
        tt=[]
        tt.append( time.time())
        ro = self.ro #1/kmax
        ko = self.ko
        G = self.G #np.log(kmax/ko)
        L = self.L #2*np.pi*N/G
        tt.append( time.time())

        bias_fft = self.bias_fft
        t_m = 2*np.pi*m/G
        alpha_m = bias_fft-1.j*t_m
        pre_factor = (ko*ro)**(-alpha_m)

        if R < 1:
            iell =  pre_factor*0.5*np.cos(np.pi*alpha_m/2)*gamma(alpha_m-2)\
                * (1/R)*((1+R)**(2-alpha_m)-(1-R)**(2-alpha_m))
            
        elif R == 1:
            iell =  pre_factor*0.5*np.cos(np.pi*alpha_m/2)*gamma(alpha_m-2)\
                * ((1+R)**(2-alpha_m))

        return iell

    def partial2(self, z1, bin_z_j, bin_lbd_j, approx=True):
        """Romberg integration of a function using scipy.integrate.romberg
        Faster and more reliable than quad used in partial

        Approximation: Put the integral_mass outside looping in m 

        TODO: Check the romberg convergence!

        """
        Z_bins = self.z_bins
        Lim_z_max = self.z_upper_limit
        Lim_z_min = self.z_lower_limit
        Z_bin_range = self.z_bin_range
        cosmo = self.cosmo
        romb_k = 6
        if (z1<=np.average(Z_bins)):    
            vec_left = np.linspace(max(Lim_z_min, z1-6*Z_bin_range),z1, 
                                    2**(romb_k-1)+1)
            vec_right = np.linspace(z1, z1+(z1-vec_left[0]),
                                    2**(romb_k-1)+1)
            vec_final = np.append(vec_left, vec_right[1:])
        else: 
            vec_right = np.linspace(z1, min(Lim_z_max, z1+6*Z_bin_range),
                                    2**(romb_k-1)+1)
            vec_left = np.linspace(z1-(vec_right[-1]-z1),z1,
                                    2**(romb_k-1)+1)
            vec_final = np.append(vec_left, vec_right[1:])

        romb_range = (vec_final[-1]-vec_final[0])/(2**romb_k)
        kernel = np.zeros(2**romb_k+1)

        if approx:
            for m in range(2**romb_k+1):
                try:
                    kernel[m] = self.dV(vec_final[m],bin_z_j)\
                                *ccl.growth_factor(cosmo, 1/(1+vec_final[m]))\
                                *self.double_bessel_integral(z1,vec_final[m])
                except Exception as ex:
                    print(f'{ex=}')
                                
            factor_approx = self.integral_mass(z1, bin_lbd_j)

        else:
            for m in range(2**romb_k+1):
                kernel[m] = self.dV(vec_final[m],bin_z_j)\
                            *ccl.growth_factor(cosmo, 1/(1+vec_final[m]))\
                            *self.double_bessel_integral(z1,vec_final[m])\
                            * self.integral_mass(vec_final[m],bin_lbd_j)
                factor_approx = 1 

        return (romb(kernel, dx=romb_range)) * factor_approx



    def double_bessel_integral(self, z1, z2):
        """Calculates the double bessel integral from I-ell algorithm, as function of z1 and z2
        """

        cosmo = self.cosmo
        ro = self.ro #1/kmax
        G = self.G #np.log(kmax/ko)
        L = self.L #2*np.pi*N/G
        r_vec = self.r_vec #np.logspace(np.log(ro), np.log(rmax), N, base=np.exp(1))
        N = self.N
        Phi_vec = self.Phi_vec #np.conjugate(np.fft.rfft(fk_vec))/L
        ko = self.ko
        bias_fft = self.bias_fft

        # definition of t, forcing it to be <= 1
        r1 = ccl.comoving_radial_distance(cosmo, 1/(1+z1))
        if z1 != z2:
            r2 = ccl.comoving_radial_distance(cosmo, 1/(1+z2))
            R = min(r1, r2)/max(r1, r2)
        else:
            r2 = r1
            R = 1
        

        I_ell_vec = [ self.I_ell(m, R )\
                     for m in range(N//2+1)]

        back_FFT_vec = np.fft.irfft(Phi_vec*I_ell_vec)*N  # FFT back
        two_fast_vec = (1/np.pi)*(ko**3)*((r_vec/ro)
                                          ** (-bias_fft))*back_FFT_vec/G

        imin = self.imin
        imax = self.imax
       
        # we will use this to interpolate the exact r(z1)
        f = interp1d(r_vec[imin:imax], 
                    two_fast_vec[imin:imax], kind='cubic') 
        try:
            return f(max(r1, r2))
        except Exception as err:
            print(err,f"""\nValue you tried to interpolate: {max(r1,r2)} Mpc, 
                Input r {r1}, {r2}
            Valid range range: [{r_vec[self.imin]}, {r_vec[self.imax]}] Mpc""")

        #CHECK THE INDEX NUMBERS
        # TODO test interpolkind
   
    @abstractmethod
    def _eval_sigma_vec(self):
        # Defer the sigma_vec to subclasses (some use inverse others use regular.)
        pass  
    
    def _get_min_radial_idx(self):
        return np.argwhere(self.r_vec  < 0.95*self.radial_lower_limit)[-1][0]
    
    def _get_max_radial_idx(self):
        return np.argwhere(self.r_vec  > 1.05*self.radial_upper_limit)[0][0]

