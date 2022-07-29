#!/usr/bin/python
import numpy as np
import os
import pytest
import tjpcov.main as cv
from tjpcov.parser import parse
import yaml
import sacc
import glob
import pyccl as ccl
import healpy as hp
import shutil


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
input_yml_ssc = os.path.join(root, "tjpcov_conf_minimal_ssc.yaml")


def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')

    if os.path.isdir('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/'):
        shutil.rmtree('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')
    os.makedirs('./tests/benchmarks/32_DES_tjpcov_bm/tjpcov_tmp/')

clean_tmp()

def get_config():
    with open(input_yml_ssc) as f:
        config = yaml.safe_load(f)
    return config


def get_halomodel_calculator_and_NFW_profile(cosmo):
    md = ccl.halos.MassDef200m()
    mf = ccl.halos.MassFuncTinker08(cosmo, mass_def=md)
    hb = ccl.halos.HaloBiasTinker10(cosmo, mass_def=md)
    cm = ccl.halos.ConcentrationDuffy08(mdef=md)

    hmc = ccl.halos.HMCalculator(cosmo, mf, hb, md)
    pNFW = ccl.halos.HaloProfileNFW(cm)

    return hmc, pNFW


def get_cl_footprint(tr1, tr2, tr3, tr4):
    config = get_config()
    mf = config['tjpcov']['mask_file']

    area = hp.nside2pixarea(32)
    m1 = hp.read_map(mf[tr1])
    m2 = hp.read_map(mf[tr2])
    m3 = hp.read_map(mf[tr3])
    m4 = hp.read_map(mf[tr4])

    m12 = m1 * m2
    m34 = m3 * m4

    alm = hp.map2alm(m12)
    blm = hp.map2alm(m34)

    cl = hp.alm2cl(alm, blm)
    cl *= (2 * np.arange(cl.size) + 1)
    cl /= np.sum(m12) * np.sum(m34) * area**2

    return cl


def get_CovarianceCalculator():
    config = get_config()
    return cv.CovarianceCalculator(config)


@pytest.mark.parametrize('tracer_comb1,tracer_comb2',
                         [(('DESgc__0', 'DESgc__0'), ('DESgc__0', 'DESgc__0')),
                          (('DESgc__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESgc__0', 'DESgc__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__0', 'DESwl__0')),
                          (('DESwl__0', 'DESwl__0'), ('DESwl__1', 'DESwl__1')),
                          ])
def test_get_SSC_cov(tracer_comb1, tracer_comb2):
    # TJPCov covariance
    cc  = get_CovarianceCalculator()
    cosmo = cc.cosmo
    s = cc.cl_data
    ell, _ = s.get_ell_cl('cl_00', 'DESgc__0', 'DESgc__0')

    ccl_tracers, _ = cc.get_tracer_info(s)
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                             tracer_comb2=tracer_comb2,
                             ccl_tracers=ccl_tracers,
                             integration_method='qag_quad',
                             include_b_modes=False)

    # Check saved file
    covf = np.load(cc.outdir + '/ssc_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                             *tracer_comb2))
    assert np.max(np.abs((covf['cov_nob'] + 1e-100) / (cov_ssc + 1e-100) - 1)) < 1e-10
    clean_tmp()

    # CCL covariance
    na = ccl.ccllib.get_pk_spline_na(cosmo.cosmo)
    a_arr, _ = ccl.ccllib.get_pk_spline_a(cosmo.cosmo, na, 0)

    bias1 = 1
    is_nc1 = False
    if 'gc' in tracer_comb1[0]:
        bias1 = cc.bias_lens[tracer_comb1[0]]
        is_nc1 = True

    bias2 = 1
    is_nc2 = False
    if 'gc' in tracer_comb1[1]:
        bias2 = cc.bias_lens[tracer_comb1[1]]
        is_nc2 = True

    bias3 = 1
    is_nc3 = False
    if 'gc' in tracer_comb2[0]:
        bias3 = cc.bias_lens[tracer_comb2[0]]
        is_nc3 = True

    bias4 = 1
    is_nc4 = False
    if 'gc' in tracer_comb2[0]:
        bias4 = cc.bias_lens[tracer_comb2[1]]
        is_nc4 = True


    hmc, prof = get_halomodel_calculator_and_NFW_profile(cosmo)
    tkk_ssc = ccl.halos.halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof,
                                                     bias1=bias1,
                                                     bias2=bias2, bias3=bias3,
                                                     bias4=bias4,
                                                     is_number_counts1=is_nc1,
                                                     is_number_counts2=is_nc2,
                                                     is_number_counts3=is_nc3,
                                                     is_number_counts4=is_nc4)

    cl_mask = get_cl_footprint(*tracer_comb1, *tracer_comb2)
    sigma2_B = ccl.sigma2_B_from_mask(cosmo, a=a_arr, mask_wl=cl_mask)

    tr1 = ccl_tracers[tracer_comb1[0]]
    tr2 = ccl_tracers[tracer_comb1[1]]
    tr3 = ccl_tracers[tracer_comb2[0]]
    tr4 = ccl_tracers[tracer_comb2[1]]
    cov_ccl = ccl.angular_cl_cov_SSC(cosmo, cltracer1=tr1, cltracer2=tr2,
                                     ell=ell, tkka=tkk_ssc,
                                     sigma2_B=(a_arr, sigma2_B),
                                     cltracer3=tr3, cltracer4=tr4)

    assert np.max(np.fabs(np.diag(cov_ssc/cov_ccl - 1))) < 1e-5
    assert np.max(np.fabs(cov_ssc/cov_ccl - 1)) < 1e-3

    # Check you get zeroed B-modes
    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=True)
    # Check saved
    assert np.max(np.abs((covf['cov'] + 1e-100) / (cov_ssc_zb + 1e-100) - 1)) < 1e-10

    ix1 = s.indices(tracers=tracer_comb1)
    ix2 = s.indices(tracers=tracer_comb2)
    ncell1 = int(ix1.size / ell.size)
    ncell2 = int(ix2.size / ell.size)

    # The covariance will have all correlations, including when EB == BE
    if (ncell1 == 3) and (tracer_comb1[0] == tracer_comb1[1]):
        ncell1 += 1
    if (ncell2 == 3) and (tracer_comb2[0] == tracer_comb2[1]):
        ncell2 += 1

    assert cov_ssc_zb.shape == (ell.size * ncell1, ell.size * ncell2)
    cov_ssc_zb = cov_ssc_zb.reshape((ell.size, ncell1, ell.size, ncell2))
    assert np.all(cov_ssc_zb[:, 0, :, 0] == cov_ssc)
    cov_ssc_zb[:, 0, :, 0] -= cov_ssc
    assert np.all(cov_ssc_zb == np.zeros_like(cov_ssc_zb))


    # Check get_SSC_cov reads file
    covf = np.load(cc.outdir + '/ssc_{}_{}_{}_{}.npz'.format(*tracer_comb1,
                                                             *tracer_comb2))
    cov_ssc = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=False)
    assert np.all(covf['cov_nob'] == cov_ssc)

    cov_ssc_zb = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                tracer_comb2=tracer_comb2,
                                ccl_tracers=ccl_tracers,
                                integration_method='qag_quad',
                                include_b_modes=True)

    assert np.all(covf['cov'] == cov_ssc_zb)


def test_get_SSC_cov_WL_with_benchmark():
    # Based on CCL benchmark test in benchmarks/test_covariances.py
    #
    # Compare against Benjamin Joachimi's code. An overview of the methodology
    # is given in appendix E.2 of 2007.01844.
    #
    # The approximation is different so one cannot expect a perfect agreement.
    # However an agreement ~5% should be enough to convince ourselves that
    # get_SSC_cov is doing what it's supposed to do.

    # Read benchmark data
    data_dir = "tests/benchmarks/SSC/"
    z, nofz = np.loadtxt(os.path.join(data_dir, "ssc_WL_nofz.txt"),
                         unpack=True)
    ell = np.loadtxt(os.path.join(data_dir, "ssc_WL_ell.txt"))
    cov_ssc_bj = np.loadtxt(os.path.join(data_dir, "ssc_WL_cov_matrix.txt"))

    # TJPCov SSC
    cc  = get_CovarianceCalculator()
    h = 0.7
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=h, n_s=0.97,
                          sigma8=0.8, m_nu=0.0)
    cc.cosmo = cosmo

    # Trick TJPCov to use the ells we want instead the ones in the file
    s = sacc.Sacc()
    s.tracers = cc.cl_data.tracers.copy()
    for dt in ['cl_ee', 'cl_eb', 'cl_be', 'cl_bb']:
        s.add_ell_cl(dt, 'DESwl__0', 'DESwl__0', ell, np.ones_like(ell))


    # Generate also a disc mask that covers fsky = 0.05. sigma2_B agrees up to
    # ~1%
    radius = 0.455
    nside = 128
    ix = hp.query_disc(nside=nside, vec=(0, -60, 0), radius=radius)
    mask = np.zeros(hp.nside2npix(nside))
    mask[ix] = 1

    # Modify tjpcov instance
    cc.cl_data = s
    cc.mask_fn['DESwl__0'] = mask

    trs = ('DESwl__0', 'DESwl__0')
    WL_tracer = ccl.WeakLensingTracer(cosmo, (z, nofz))
    cov_ssc = cc.get_SSC_cov(trs, trs, {'DESwl__0': WL_tracer},
                             include_b_modes=False)

    clean_tmp()
    # Tests
    var_ssc_ccl = np.diag(cov_ssc)
    off_diag_1_ccl = np.diag(cov_ssc, k=1)

    # At large scales, CCL uses a different convention for the Limber
    # approximation. This factor accounts for this difference
    ccl_limber_shear_fac = np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+1/2)**2
    cov_ssc_bj_corrected = cov_ssc_bj * np.outer(ccl_limber_shear_fac**2,
                                                 ccl_limber_shear_fac**2)
    var_bj = np.diag(cov_ssc_bj_corrected)
    off_diag_1_bj = np.diag(cov_ssc_bj_corrected, k=1)


    # TODO: After the refactoring, and freeing some of the halo model choices
    # for the SSC, we should update this test to a greater accuracy.
    #
    # Note: the array of a in the CCL test: a = np.linspace(1/(1+6), 1, 100)
    #
    #   - If I use the halomod_Tk3D_SSC, the same mass function
    #   (MassFuncTinker10, instead of MassFuncTinker08, as is currently in
    #   tjpcov/main.py) and the array of a in the CCL test, the test passes
    #   with the same precision as in CCL (<3%)
    #  - If I change the mass function to MassFuncTinker08, the error goes up
    #  to 6.8%
    #  - If I use the halomod_Tk3D_SSC_linear_bias, MassFuncTinker10 and the
    #  same a's as in the test, the error goes to ~6%. (This is the error
    #  introduced by the approximation, then)
    #  - If I use the halomod_Tk3D_SSC_linear_bias, MassFuncTinker08 and the
    #  same a's as in the test, the test passes with <3% error
    #  - If I use the a's as implemented in tjpcov/main.py (i.e. from the pk
    #  splines): the error goes up to ~18% irrespectively of the Tk3D, or mass
    #  function that one choses. That's why the threshold is 0.2
    #
    # This error instability seem to appear only on the first ell. So I will do
    # the test without including the first column & row of the covariance.

    assert np.max(np.fabs(var_ssc_ccl/var_bj - 1)[1:]) < 0.07
    assert np.max(np.fabs(off_diag_1_ccl/off_diag_1_bj - 1)[1:]) < 0.07
    assert np.max(np.fabs(cov_ssc/cov_ssc_bj_corrected - 1)[1:][:, 1:]) < 0.07


def test_get_all_cov_SSC():
    # Smoke
    cc  = get_CovarianceCalculator()
    cov_ssc = cc.get_all_cov_SSC() + 1e-100

    # Check the covariance has the right terms for 00, 0e and ee
    s = cc.cl_data.copy()
    s.covariance.covmat = cov_ssc
    s.remove_selection(data_type='cl_0b')
    s.remove_selection(data_type='cl_eb')
    s.remove_selection(data_type='cl_be')
    s.remove_selection(data_type='cl_bb')

    trs = s.get_tracer_combinations()
    ccl_tracers, _ = cc.get_tracer_info(s)
    for tracer_comb1 in trs:
        i1 = s.indices(tracers=tracer_comb1)
        for tracer_comb2 in trs:
            i2 = s.indices(tracers=tracer_comb2)
            cov_ssc2 = cc.get_SSC_cov(tracer_comb1=tracer_comb1,
                                      tracer_comb2=tracer_comb2,
                                      ccl_tracers=ccl_tracers,
                                      integration_method='qag_quad',
                                      include_b_modes=False) + 1e-100

            cov_i1i2 = s.covariance.covmat[i1][:, i2]
            assert np.max(np.abs(cov_i1i2 / cov_ssc2 - 1)) < 1e-5

    # Check the covariance is 0 for all the b-modes
    s = cc.cl_data.copy()
    s.covariance.covmat = cov_ssc
    s.remove_selection(data_type='cl_00')
    s.remove_selection(data_type='cl_0e')
    s.remove_selection(data_type='cl_ee')

    assert np.all(s.covariance.covmat == 1e-100)

clean_tmp()
