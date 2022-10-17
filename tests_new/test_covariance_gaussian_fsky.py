#!/usr/bin/python3
import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new import bin_cov
from tjpcov_new.covariance_gaussian_fsky import \
    CovarianceFourierGaussianFsky, CovarianceRealGaussianFsky
from tjpcov_new.covariance_io import CovarianceIO
import yaml
import healpy as hp
import sacc
import shutil

# INPUT
# CCL and sacc input:
os.makedirs('tests/tmp/', exist_ok=True)
cosmo_filename = "tests/data/cosmo_desy1.yaml"
cosmo = ccl.Cosmology.read_yaml(cosmo_filename)
with open("tests/tmp/cosmos_desy1_v2p1p0.pkl", 'wb') as ff:
    pickle.dump(cosmo, ff)

# SETUP
input_yml = "tests_new/data/conf_tjpcov_minimal.yaml"
input_yml_real = "tests_new/data/conf_tjpcov_minimal_real.yaml"
cfsky = CovarianceFourierGaussianFsky(input_yml)
cfsky_real = CovarianceRealGaussianFsky(input_yml_real)
ccl_tracers, tracer_Noise = cfsky.get_tracer_info()


def clean_tmp():
    if os.path.isdir('./tests/tmp'):
        shutil.rmtree('./tests/tmp/')
    os.makedirs('./tests/tmp')


def get_config():
    return CovarianceIO(input_yml).config


def test_smoke():
    cfsky = CovarianceFourierGaussianFsky(input_yml)
    # cfsky = CovarianceRealGaussianFsky(input_yml)

    # Check it raises an error if fsky is not given
    config = get_config()
    config['GaussianFsky'] = {}
    with pytest.raises(ValueError):
        cfsky = CovarianceFourierGaussianFsky(config)


def test_Fourier_get_binning_info():
    cfsky = CovarianceFourierGaussianFsky(input_yml)
    ell, ell_eff, ell_edges = cfsky.get_binning_info()

    assert np.all(ell_eff == cfsky.get_ell_eff())
    assert np.allclose((ell_edges[1:]+ell_edges[:-1])/2, ell_eff)

    with pytest.raises(NotImplementedError):
        cfsky.get_binning_info('log')


def test_Fourier_get_covariance_block():
    # Test made independent of pickled objects
    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')

    s = cfsky.io.sacc_file

    ell, ell_bins, ell_edges = cfsky.get_binning_info()
    ccltr = ccl_tracers['lens0']
    cl = ccl.angular_cl(cosmo, ccltr, ccltr, ell) + tracer_Noise['lens0']

    fsky = cfsky.fsky
    dl = np.gradient(ell)
    cov = np.diag(2 * cl**2 / ((2 * ell + 1) * fsky * dl))
    lb, cov = bin_cov(r=ell, r_bins=ell_edges, cov=cov)

    gcov_cl_1 = cfsky.get_covariance_block(tracer_comb1=tracer_comb1,
                                           tracer_comb2=tracer_comb2,
                                           include_b_modes=False)
    np.testing.assert_allclose(gcov_cl_1, cov)

    trs = ('src0', 'src0')
    gcov_cl_1 = cfsky.get_covariance_block(tracer_comb1=trs,
                                           tracer_comb2=trs,
                                           include_b_modes=False)
    gcov_cl_1b = cfsky.get_covariance_block(tracer_comb1=trs,
                                           tracer_comb2=trs,
                                           include_b_modes=True)

    nbpw = lb.size
    assert np.all(gcov_cl_1b[:nbpw][:, :nbpw] == gcov_cl_1)
    gcov_cl_1b = gcov_cl_1b.reshape((nbpw, 4, nbpw, 4), order='F')
    gcov_cl_1b[:, 0, :, 0] -= gcov_cl_1
    assert not np.any(gcov_cl_1b)


@pytest.mark.parametrize('tracer_comb1',
                          [('lens0', 'lens0'),
                           ('src0', 'lens0'),
                           ('lens0', 'src0'),
                           ('src0', 'src0'),
                          ])
@pytest.mark.parametrize('tracer_comb2',
                          [('lens0', 'lens0'),
                           ('src0', 'lens0'),
                           ('lens0', 'src0'),
                           ('src0', 'src0'),
                          ])
def test_Real_get_fourier_block(tracer_comb1, tracer_comb2):
    cov = cfsky_real._get_fourier_block(tracer_comb1, tracer_comb2)
    cov2 = cfsky.get_covariance_block(tracer_comb1, tracer_comb2,
                                      for_real=True, lmax=cfsky_real.lmax)

    norm = np.pi * 4 * cfsky_real.fsky
    assert np.all(cov == cov2 / norm)


def test_smoke_get_covariance():
    # Check that we can get the full covariance
    cfsky.get_covariance()
    # Real test commented out because we don't have a method to build the full
    # covariance atm
    # cfsky_real.get_covariance()
