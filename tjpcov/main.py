import pdb
from tjpcov import wigner_transform, bin_cov, parse
import numpy as np
import sacc
import pyccl as ccl
import sys
import os
import inspect 

cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd)+"/tjpcov")


d2r = np.pi/180


class CovarianceCalculator():
    def __init__(self, 
        do_xi=False,
        sacc_file = None, # FIXME or cl_file or sacc_file or xi_file
        xi_file=None,
        cl_file=None,
        cosmo=None,
        cov_type = 'gauss',
        mask_file =None,
        fsky = 1,
        ngal_lens=None,
        ngal_src=None,
        sigma_e=None,
        bias_lens=None,
        IA=0.5):
        """
        Covariance Calculator object for TJPCov. 

        .. note::
            - the cosmo_fn parameter is always necessary  
            - theta input in arcmin
            - Is reading all config from a single yaml a good option?
            - sacc passing values after scale cuts and no angbin_edges
            - angbin_edges necessary for bin averages
            - Check firecrown's way to handle survey features

        Parameters
        ----------
        do_xi : bool
            Set it False (default) to evaluate the Harmonic Space covariance. 
            Otherwise, it will produce the configuration space covariance.
        xi_file : sacc
            This file keeps the dndz used by tjpcov.
        cl_file : sacc
            This file keeps the dndz used by tjpcov.
        cosmo : str
            This option will handle the cosmology through cosmo=ccl.Cosmology() 
            object. Valid options are: 
            - A path to a cosmo object
            - A path to yaml file produced by CCL.
        cov_type : str
            (default `gauss`) kind of covariance evaluated. Valid options are
            `gauss` for gaussian covariance and `nmt` for NaMASTER covariance
            in Harmonic Space.
        mask_file : str
            (optional) Path to the mask file. Requested by cov_type = 'nmt'
        fsky : float
            Fraction observed of the sky covered by the footprint.
        ngal_lens : list
            List with the ordered values for number density of lens galaxies. 
            in Ngal/arcmin**2.
        ngal_src : list
            List with the ordered values for number density of source galaxies 
            in Ngal/arcmin**2.
        sigma_e : list
            List with the ordered values for sigma_epsilon for sources.
        bias_lens : list
            List with the ordered values for linear bias of lenses.
        IA : float 
            Intrisic Alignment (default=0.5)

        Results
        -------
        
    
        """
        self.sacc_file=sacc_file
        self.do_xi=do_xi
        self.xi_file = xi_file
        self.cl_file = cl_file
        # self.cosmo = cosmo
        self.cov_type=cov_type
        self.mask_file = mask_file
        self.fsky = fsky
        # self.ngal_lens = ngal_lens
        # self.ngal_src = ngal_src
        # self.sigma_e = sigma_e
        # self.bias_lens = bias_lens
        self.IA = IA


        cosmo_fn = cosmo

        self.bias_lens = {f'lens{i}': v for i,v in enumerate(bias_lens) }
        #= {k.replace('bias_',''):v for k,v in config['tjpcov'].items() 
                         #   if 'bias_lens' in k}

        # self.IA = config['tjpcov'].get('IA')
        if ngal_lens is not None:
            self.Ngal = {f'lens{i}': v*3600/d2r**2 for i,v in enumerate(ngal_lens)  }
                        #{k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items() 
                         #   if 'Ngal' in k}
        if ngal_src is not None:                         
            Ngal_src = {f'src{i}': v*3600/d2r**2 for i,v in enumerate(ngal_src)}
            self.Ngal.update(Ngal_src)
        #{k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items() 
        #   
        if sigma_e is not None:               #   if 'Ngal_src' in k}
            self.sigma_e = {f'src{i}': v for i,v in enumerate(sigma_e)} 
        # {k.replace('sigma_e','src'):v for k, v in config['tjpcov'].items() 
        #                     if 'sigma_e' in k}
        

        # Treating fsky = 1 if no input is given
        # self.fsky = config['tjpcov'].get('fsky')
        

        ## config, inp_dat = parse(tjpcov_cfg)

        ## self.do_xi = config['tjpcov'].get('do_xi')

        if not isinstance(self.do_xi, bool):
            raise Exception("Err: check if you set do_xi: False (Harmonic Space) "
                            + "or do_xi: True in 'tjpcov' field of your yaml")

        print("Starting TJPCov covariance calculator for", end=' ')
        print("Configuration space" if self.do_xi else "Harmonic Space")

        if self.do_xi:
            xi_fn = xi_file
        else:
            cl_fn = cl_file

        # cosmo_fn = cosmo_fn#config['tjpcov'].get('cosmo')
        # sacc_fn  = config['tjpcov'].get('sacc_file')

        # biases
        # reading values w/o passing the number of tomographic bins
        # import pdb; pdb.set_trace()
        # self.bias_lens = {k.replace('bias_',''):v for k,v in config['tjpcov'].items() 
        #                     if 'bias_lens' in k}
        # self.IA = config['tjpcov'].get('IA')
        # self.Ngal = {k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items() 
        #                     if 'Ngal' in k}
        # # self.Ngal_src = {k.replace('Ngal_',''):v*3600/d2r**2 for k, v in config['tjpcov'].items() 
        # #                     if 'Ngal_src' in k}
        # self.sigma_e = {k.replace('sigma_e','src'):v for k, v in config['tjpcov'].items() 
        #                     if 'sigma_e' in k}
        

        # # Treating fsky = 1 if no input is given
        # self.fsky = config['tjpcov'].get('fsky')
        if self.fsky is None:
            print("No input for fsky. Assuming ", end='')
            self.fsky=1

        print(f"fsky={self.fsky}")
        

        if cosmo_fn is None or cosmo_fn == 'set':
            self.cosmo = self.set_ccl_cosmo(config)

        elif cosmo_fn.split('.')[-1] == 'yaml':
            self.cosmo = ccl.Cosmology.read_yaml(cosmo_fn)
            # TODO: remove this hot fix of ccl
            self.cosmo.config.transfer_function_method = 1

        elif cosmo_fn.split('.')[-1]  == 'pkl':
            import pickle
            with open(cosmo_fn, 'rb') as ccl_cosmo_file:
                self.cosmo = pickle.load(ccl_cosmo_file)


        elif isinstance(cosmo_fn, ccl.core.Cosmology):  
            self.cosmo = cosmo_fn
        else:
            raise Exception(
                "Err: File for cosmo field in input not recognized")

        # TO DO: remove this hotfix
        self.xi_data, self.cl_data = None, None
        # pdb.set_trace()
        if self.do_xi:
            self.xi_data = sacc.Sacc.load_fits(sacc_file)
                # config['tjpcov'].get('sacc_file'))

        # TO DO: remove this dependence here
        #elif not do_xi: 
        self.cl_data = sacc.Sacc.load_fits(cl_file)
            # config['tjpcov'].get('cl_file'))
        # TO DO: remove this dependence here
        ell_list = self.get_ell_theta(self.cl_data,  # fix this
                                      'galaxy_density_cl',
                                      ('lens0', 'lens0'),
                                      'linear', do_xi=False)

        self.mask_fn = mask_file #config['tjpcov'].get('mask_file')  # windown handler TBD

        # fao Set this inside get_ell_theta ?
        # ell, ell_bins, ell_edges = None, None, None
        theta, theta_bins, theta_edges = None, None, None



        # fix this for getting from the sacc file:
        th_list = self.set_ell_theta(2.5, 250., 20, do_xi=True)

        self.theta,  self.theta_bins, self.theta_edges,  = th_list


        # ell is the value for WT
        self.ell, self.ell_bins, self.ell_edges = ell_list

        # Calling WT in method, only if do_xi
        self.WT = None

        return

    @classmethod        
    def from_yaml(cls, tjpcov_cfg):
        """
        This method reads input from yaml 
    
        Parameters
        ----------
        tjpcov_cfg (str): 
            filename and path to the TJPCov configuration yaml
            Check minimal example at: tests/data/conf_tjpcov_minimal.yaml
            This file MUST have 
            - a sacc path
            - a xi_fn OR cl_fn
            - a reference to ccl.Cosmology object (in pickle format)  
                OR cosmology.yaml file generated by CCL
            - ...
            it contains info from the deprecated files:

            cosmo_fn(pyccl.object or str):
                Receives the cosmo object or a the yaml filename
                WARNING CCL Cosmo write_yaml seems to not pass
                        the transfer_function
            sacc_fn_xi/cl (None, str):
                path to sacc file yaml
                sacc object containing the tracers and binning to be used 
                in covariance calculation
            mask (None, dict, str):
                If None, used window function specified in the sacc file
                if dict, the keys are sacc tracer names and values are 
                either HealSparse inverse variance
                maps 
                if float it is assumed to be f_sky value
        """
        config, inp_dat = parse(tjpcov_cfg)
        params = config['tjpcov']

        do_xi = params.get('do_xi')

        cosmo_fn = params.get('cosmo')

        fsky = params.get('fsky')

        # -----------------
        # lists
        # -----------------
        bias_lens = [v for k,v in params.items() 
                            if 'bias_lens' in k]
        IA_ = params.get('IA')

        keys = params.keys()

        has_ngal_lens, has_ngal_src, has_bias_lens, has_sigma_e = [False]*4
        for k in keys:
            if 'ngal_lens' in k.lower(): has_ngal_lens = True
            if 'ngal_src' in k.lower(): has_ngal_src = True 
            if 'bias_lens' in k.lower(): has_bias_lens = True 
            if 'sigma_e' in k.lower(): has_sigma_e = True 

        mask_fn = params.get('mask_file')  # windown handler

        init_params_names = inspect.signature(cls).parameters.keys()

        inits = {k: params[k] for k in init_params_names if k in params}
        if has_ngal_lens:
            inits.update(dict(ngal_lens = [v for i,v in params.items() 
                                            if 'Ngal_lens' in i ]))
        if has_ngal_src:
            inits.update(dict(ngal_src = [v for i,v in params.items() 
                                            if 'Ngal_src' in i ]))
        if has_bias_lens:
            inits.update(dict(bias_lens = [v for i,v in params.items() 
                                            if 'bias_lens' in i ]))
        if has_sigma_e:
            inits.update(dict(sigma_e = [v for i,v in params.items() 
                                        if 'sigma_e' in i ]))

        return cls(**inits) 

    def print_setup(self, output=None):
        """
        Placeholder of function to return setup
        TODO: Check the current setup for TJPCovs
        """
        cosmo = self.cosmo
        ell = self.ell
        if self.do_xi:
            bins = self.theta_bins
        else:
            bins = self.ell_bins            
        run_configuration = {
        'do_xi': self.do_xi,
        'bins': bins
        }
        # TODO: save as yaml output
        if isinstance(output, str):
            with open(output, 'w') as ff:
                ff.write('....txt')


    def set_ccl_cosmo(self, config):
        """
        set the ccl cosmo from paramters in config file

        """
        print("Setting up cosmology...")

        cosmo_param_names = ['Omega_c', 'Omega_b', 'h',
                             'sigma8', 'n_s', 'transfer_function']
        cosmo_params = {name: config['parameters'][name]
                        for name in cosmo_param_names}
        cosmo = ccl.Cosmology(**cosmo_params)
        return cosmo


    def set_ell_theta(self, ang_min, ang_max, n_ang,
                      ang_scale='linear', do_xi=False):
        """
        Utility for return custom theta/ell bins (outside sacc)

        Parameters:
        -----------
        ang_min (int, float):
            if do_xi, ang is assumed to be theta (arcmin)
            if do_xi == False,  ang is assumed to be ell 
        Returns:
        --------
            (theta, theta_edges ) (degrees):
        """
        # FIXME:
        # Use sacc is passing this
        if not do_xi:
            ang_delta = (ang_max-ang_min)//n_ang
            ang_edges = np.arange(ang_min, ang_max+1, ang_delta)
            ang = np.arange(ang_min, ang_max + ang_delta - 2)

        if do_xi:
            th_min = ang_min/60  # in degrees
            th_max = ang_max/60
            n_th_bins = n_ang
            ang_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                    n_th_bins+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), n_th_bins*30)
            # binned covariance can be sensitive to the th values. Make sure
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, n_th_bins*30)

            ang = np.unique(np.sort(np.append(th, th2)))
            ang_bins = 0.5 * (ang_edges[1:] + ang_edges[:-1])

            return ang, ang_bins, ang_edges  # TODO FIXIT

        return ang, ang_edges


    def get_ell_theta(self, two_point_data, data_type, tracer_comb, ang_scale,
                      do_xi=False):
        """
        Get ell or theta for bins given the sacc object 
        For now, presuming only log and linear bins 

        Parameters:
        -----------

        Returns:
        --------
        """
        ang_name = "ell" if not do_xi else 'theta'

        # assuming same ell for all bins:
        data_types = two_point_data.get_data_types()
        ang_bins = two_point_data.get_tag(ang_name, data_type=data_types[0],
                                          tracers=tracer_comb)

        ang_bins = np.array(ang_bins)

        angb_min, angb_max = ang_bins.min(), ang_bins.max()
        if ang_name == 'theta':
            # assuming log bins
            del_ang = (ang_bins[1:]/ang_bins[:-1]).mean()
            ang_scale = 'log'
            assert 1 == 1

        elif ang_name == 'ell':
            # assuming linear bins
            del_ang = (ang_bins[1:] - ang_bins[:-1])[0]
            ang_scale = 'linear'

        ang, ang_edges = self.set_ell_theta(angb_min-del_ang/2,
                                            angb_max+del_ang/2,
                                            len(ang_bins),
                                            ang_scale=ang_scale, do_xi=do_xi)
        # Sanity check
        if ang_scale == 'linear':
            assert np.allclose((ang_edges[1:]+ang_edges[:-1])/2, ang_bins), \
                "differences in produced ell/theta"
        return ang, ang_bins, ang_edges


    def wt_setup(self, ell, theta):
        """
        Set this up once before the covariance evaluations

        Parameters:
        -----------
        ell (array): array of multipoles
        theta ( array): array of theta in degrees

        Returns:
        --------
        """
        # ell = two_point_data.metadata['ell']
        # theta_rad = two_point_data.metadata['th']*d2r
        # get_ell_theta()

        WT_factors = {}
        WT_factors['lens', 'source'] = (0, 2)
        WT_factors['source', 'lens'] = (2, 0)  # same as (0,2)
        WT_factors['source', 'source'] = {'plus': (2, 2), 'minus': (2, -2)}
        WT_factors['lens', 'lens'] = (0, 0)

        self.WT_factors = WT_factors

        ell = np.array(ell)
        if not np.alltrue(ell > 1):
            # fao check warnings in WT for ell < 2
            print("Removing ell=1 for Wigner Transformation")
            ell = ell[(ell > 1)]

        WT_kwargs = {'l': ell,
                     'theta': theta*d2r,
                     's1_s2': [(2, 2), (2, -2), (0, 2), (2, 0), (0, 0)]}

        WT = wigner_transform(**WT_kwargs)
        return WT


    def get_cov_WT_spin(self, tracer_comb=None):
        """
        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------
        WT_factors: 

        """
    #     tracers=tuple(i.split('_')[0] for i in tracer_comb)
        tracers = []
        for i in tracer_comb:
            if 'lens' in i:
                tracers += ['lens']
            if 'src' in i:
                tracers += ['source']
        return self.WT_factors[tuple(tracers)]


    def get_tracer_info(self, two_point_data={}):
        """
        Creates CCL tracer objects and computes the noise for all the tracers
        Check usage: Can we call all the tracer at once?

        Parameters:
        -----------
            two_point_data (sacc obj):

        Returns:
        --------
            ccl_tracers: dict, ccl obj
                ccl.WeakLensingTracer or ccl.NumberCountsTracer
            tracer_Noise ({dict: float}): 
                shot (shape) noise for lens (sources)
        """
        ccl_tracers = {}
        tracer_Noise = {}
        # b = { l:bi*np.ones(len(z)) for l, bi in self.lens_bias.items()}

        for tracer in two_point_data.tracers:
            tracer_dat = two_point_data.get_tracer(tracer)
            z = tracer_dat.z

            # FIXME: Following should be read from sacc dataset.--------------
            #Ngal = 26.  # arc_min^2
            #sigma_e = .26
            #b = 1.5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
            # AI = .5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
            #Ngal = Ngal*3600/d2r**2
            # ---------------------------------------------------------------

            dNdz = tracer_dat.nz
            dNdz /= (dNdz*np.gradient(z)).sum()
            dNdz *= self.Ngal[tracer]
            #FAO  this should be called by tomographic bin
            if 'source' in tracer or 'src' in tracer:
                IA_bin = self.IA*np.ones(len(z)) # fao: refactor this
                ccl_tracers[tracer] = ccl.WeakLensingTracer(
                    self.cosmo, dndz=(z, dNdz), ia_bias=(z, IA_bin))
                # CCL automatically normalizes dNdz
                tracer_Noise[tracer] = self.sigma_e[tracer]**2/self.Ngal[tracer]

            elif 'lens' in tracer:
                # import pdb; pdb.set_trace()
                b = self.bias_lens[tracer] * np.ones(len(z))
                tracer_Noise[tracer] = 1./self.Ngal[tracer]
                ccl_tracers[tracer] = ccl.NumberCountsTracer(
                    self.cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))
        return ccl_tracers, tracer_Noise

    # Outter class     
    def cl_gaussian_cov(self, tracer_comb1=None, tracer_comb2=None,
                        ccl_tracers=None, tracer_Noise=None,
                        two_point_data=None, do_xi=False,
                        xi_plus_minus1='plus', xi_plus_minus2='plus'):
        """
        Compute a single covariance matrix for a given pair of C_ell or xi

        Returns:
        --------
            final:  unbinned covariance for C_ell
            final_b : binned covariance 
        """
        # fsky should be read from the sacc
        # tracers 1,2,3,4=tracer_comb1[0],tracer_comb1[1],tracer_comb2[0],tracer_comb2[1]
        # ell=two_point_data.metadata['ell']
        # fao to discuss: indices
        cosmo = self.cosmo
        #do_xi=self.do_xi

        if not do_xi:
            ell = self.ell
        else:
            # FIXME:  check the max_ell here in the case of only xi
            ell = self.ell

        cl = {}
        cl[13] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell)
        cl[24] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell)
        cl[14] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell)
        cl[23] = ccl.angular_cl(
            cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell)

        SN = {}
        SN[13] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[0] else 0
        SN[24] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[1] else 0
        SN[14] = tracer_Noise[tracer_comb1[0]
                              ] if tracer_comb1[0] == tracer_comb2[1] else 0
        SN[23] = tracer_Noise[tracer_comb1[1]
                              ] if tracer_comb1[1] == tracer_comb2[0] else 0

        if do_xi:
            norm = np.pi*4* self.fsky #two_point_data.metadata['fsky']
        else:  # do c_ell
            norm = (2*ell+1)*np.gradient(ell)* self.fsky #two_point_data.metadata['fsky']

        coupling_mat = {}
        coupling_mat[1324] = np.eye(len(ell))  # placeholder
        coupling_mat[1423] = np.eye(len(ell))  # placeholder

        cov = {}
        cov[1324] = np.outer(cl[13]+SN[13], cl[24]+SN[24])*coupling_mat[1324]
        cov[1423] = np.outer(cl[14]+SN[14], cl[23]+SN[23])*coupling_mat[1423]

        cov['final'] = cov[1423]+cov[1324]

        if do_xi:
            if self.WT is None:  # class modifier of WT initialization
                print("Preparing WT...")
                self.WT = self.wt_setup(self.ell, self.theta)
                print("Done!")

            # Fixme: SET A CUSTOM ELL FOR do_xi case, in order to use
            # a single sacc input filefile
            ell = self.ell
            s1_s2_1 = self.get_cov_WT_spin(tracer_comb=tracer_comb1)
            s1_s2_2 = self.get_cov_WT_spin(tracer_comb=tracer_comb2)
            if isinstance(s1_s2_1, dict):
                s1_s2_1 = s1_s2_1[xi_plus_minus1]
            if isinstance(s1_s2_2, dict):
                s1_s2_2 = s1_s2_2[xi_plus_minus2]
            th, cov['final'] = self.WT.projected_covariance2(l_cl=ell, s1_s2=s1_s2_1,
                                                             s1_s2_cross=s1_s2_2,
                                                             cl_cov=cov['final'])

        cov['final'] /= norm

        if do_xi:
            thb, cov['final_b'] = bin_cov(
                r=th/d2r, r_bins=self.theta_edges, cov=cov['final'])
            # r=th/d2r, r_bins=two_point_data.metadata['th_bins'], cov=cov['final'])
        else:
            # if two_point_data.metadata['ell_bins'] is not None:
            if self.ell_edges is not None:
                lb, cov['final_b'] = bin_cov(
                    r=self.ell, r_bins=self.ell_edges, cov=cov['final'])
                # r=ell, r_bins=two_point_data.metadata['ell_bins'], cov=cov['final'])

    #     cov[1324]=None #if want to save memory
    #     cov[1423]=None #if want to save memory
        return cov

    def get_all_cov(self, do_xi=False):
        """
        Compute all the covariances and then combine them into one single giant matrix
        Parameters:
        -----------
        two_point_data (sacc obj): sacc object containg two_point data

        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations. 
            Npt = (number of bins ) * (number of combinations)

        """
        # FIXME: Only input needed should be two_point_data,
        # which is the sacc data file. Other parameters should be
        # included within sacc and read from there."""

        two_point_data = self.xi_data if do_xi else self.cl_data

        ccl_tracers, tracer_Noise = self.get_tracer_info(
            two_point_data=two_point_data)

        # we will loop over all these
        tracer_combs = two_point_data.get_tracer_combinations()
        N2pt = len(tracer_combs)

        N_data = len(two_point_data.indices())
        print(f"Producing covariance with {N_data}x{N_data} points", end=" ")
        print(f"({N2pt} combinations of tracers)")

        # if two_point_data.metadata['ell_bins'] is not None:
        #     Nell_bins = len(two_point_data.metadata['ell_bins'])-1
        # else:
        #     Nell_bins = len(two_point_data.metadata['ell'])

        # if do_xi:
        #     Nell_bins = len(two_point_data.metadata['th_bins'])-1

        # cov_full = np.zeros((Nell_bins*N2pt, Nell_bins*N2pt))

        cov_full = np.zeros((N_data, N_data))

        # Fix this loop for uneven scale cuts (different N_ell)
        for i in np.arange(N2pt):
            print("{}/{}".format(i+1, N2pt))
            tracer_comb1 = tracer_combs[i]
            # solution for non-equal number of ell in bins
            Nell_bins_i = len(two_point_data.indices(tracers=tracer_comb1))
            indx_i = i*Nell_bins_i
            for j in np.arange(i, N2pt):
                tracer_comb2 = tracer_combs[j]
                Nell_bins_j = len(two_point_data.indices(tracers=tracer_comb2))
                indx_j = j*Nell_bins_j
                cov_ij = self.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                              tracer_comb2=tracer_comb2,
                                              ccl_tracers=ccl_tracers,
                                              tracer_Noise=tracer_Noise,
                                              do_xi=do_xi,
                                              two_point_data=two_point_data)

                # if do_xi or two_point_data.metadata['ell_bins'] is not None:
                # check
                if do_xi or self.ell_bins is not None:
                    cov_ij = cov_ij['final_b']
                else:
                    cov_ij = cov_ij['final']

                cov_full[indx_i:indx_i+Nell_bins_i,
                         indx_j:indx_j+Nell_bins_j] = cov_ij
                cov_full[indx_j:indx_j+Nell_bins_i,
                         indx_i:indx_i+Nell_bins_j] = cov_ij.T
        return cov_full

    def create_sacc_cov(output, do_xi=False):
        """ Write created cov to a new sacc object

        Parameters:
        ----------
        output (str): filename output
        do_xi (bool): do_xi=True for real space, do_xi=False for harmonic
            space

        Returns:
        -------
        None

        """
        print("Placeholder...")
        if do_xi:
            print(f"Saving xi covariance as \n{output}")
        else:
            print(f"Saving xi covariance as \n{output}")
        pass


if __name__ == "__main__":
    import tjpcov.main as cv
    import pickle
    import sys
    import os
    

    cwd = os.getcwd()
    sys.path.append(os.path.dirname(cwd)+"/tjpcov")
    # reference:
    with open(f"./tests/data/tjpcov_cl.pkl", "rb") as ff:
        cov0cl = pickle.load(ff)

    # pdb.set_trace()
    tjp0 = cv.CovarianceCalculator.read_yaml(tjpcov_cfg=f"{cwd}/tests/data/conf_tjpcov_minimal.yaml")
    
    ccl_tracers, tracer_Noise = tjp0.get_tracer_info(tjp0.cl_data)
    trcs = tjp0.cl_data.get_tracer_combinations()

    gcov_cl_0 = tjp0.cl_gaussian_cov(tracer_comb1=('lens0', 'lens0'),
                                     tracer_comb2=('lens0', 'lens0'),
                                     ccl_tracers=ccl_tracers,
                                     tracer_Noise=tracer_Noise,
                                     two_point_data=tjp0.cl_data,
                                     )


    if np.array_equal(gcov_cl_0['final_b'].diagonal()[:], cov0cl.diagonal()[:24]):
        print("Cov (diagonal):\n", gcov_cl_0['final_b'].diagonal()[:])
    else:
        print(gcov_cl_0['final_b'].diagonal()[:], cov0cl.diagonal()[:24])
    
    