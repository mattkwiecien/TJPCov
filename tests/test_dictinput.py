# Testing minimal yaml configuration

import numpy as np
import tjpcov.main as cv
from tjpcov import wigner_transform, bin_cov, parse
import sacc
import pyccl as ccl
import pytest
import pickle
import pdb

# Remove it after the setup.py
import sys
import os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd)+"/tjpcov")



def depickling(name):
    """get the target data 
    """
    try:
        with open(f"tests/val_{name}.pkl", 'rb') as ff:
            return pickle.load(ff)
    except:
        print(f"missing {name}")
        print(os.listdir("tests/"))


def check_numerr(a, b, f=np.array_equal):
    """ wrapper for test
    """
    print('ok' if f(a, b) else 'Check for numerical errors')


# INPUT
# CCL and sacc input:
with open("tests/data/cosmos_desy1_v2p2p0.pkl", 'rb') as ff:
    cosmo = pickle.load(ff)

cosmo_filename = "tests/data/cosmo_desy1.yaml"

xi_fn = "examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
cl_fn = "examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits"

# Reference Output extracted from ipynb:
with open("tests/data/tjpcov_cl.pkl", "rb") as ff:
    ref_cov0cl = pickle.load(ff)
with open("tests/data/tjpcov_xi.pkl", "rb") as ff:
    ref_cov0xi = pickle.load(ff)

ref_covnobin = depickling("covnobin")  # ell datavectors bins #EDGES
ref_md_ell_bins = depickling('metadata_ell_bins')
ref_thb = depickling('thb')
ref_WT = depickling('WT')  # Wigner Transform object
ref_md_th = depickling('metadata_th')
ref_md_th_bins = depickling('metadata_th_bins')  # th datavectors bins #EDGES
ref_md_ell_bins = depickling('metadata_ell_bins')

# SETUP
#tjp0 = cv.CovarianceCalculator(tjpcov_cfg="tests/data/conf_tjpcov_minimal.yaml")


tjp0 = cv.CovarianceCalculator.from_yaml("tests/data/conf_tjpcov_minimal.yaml")

input_dictionary = { 'IA': 0.5,
                     'bias_lens': [1.5, 1.5, 1.5, 1.5, 1.5],
                     'cl_file': 'examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits',
                     'cosmo': 'tests/data/cosmos_desy1_v2p2p0.pkl',
                     'cov_type': 'gauss',
                     'do_xi': False,
                     'fsky': 0.3,
                     'mask_file': None,
                     'ngal_lens': [26, 26, 26, 26, 26],
                     'ngal_src': [26, 26, 26, 26],
                     'sacc_file': 'examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits',
                     'sigma_e': [0.26, 0.26, 0.26, 0.26],
                     'xi_file': 'examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits'}


input_dictionary_cs = dict(input_dictionary)
input_dictionary_cs.update({"do_xi":True})

tjp1 = cv.CovarianceCalculator(**input_dictionary)
tjp1_cs = cv.CovarianceCalculator(**input_dictionary_cs)

tracer_comb1 = ('lens0', 'lens1')
tracer_comb2 = ('src0', 'src1')

ccl_tracers, tracer_Noise = tjp0.get_tracer_info(tjp0.cl_data)
gcov_cl_1 = tjp0.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                 tracer_comb2=tracer_comb2,
                                 ccl_tracers=ccl_tracers,
                                 tracer_Noise=tracer_Noise,
                                 two_point_data=tjp0.cl_data,
                                 do_xi=False)

ccl_tracers, tracer_Noise = tjp1.get_tracer_info(tjp1.cl_data)

gcov_cl_1_dict = tjp1.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                      tracer_comb2=tracer_comb2,
                                      ccl_tracers=ccl_tracers,
                                      tracer_Noise=tracer_Noise,
                                      two_point_data=tjp1.cl_data,
                                      do_xi=False)

def test_dictionary():
    np.testing.assert_allclose(gcov_cl_1['final_b'],
                               gcov_cl_1_dict['final_b'])


def test_cl_block():
    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')
    print(f"Checking covariance block. \
          Tracer combination {tracer_comb1} {tracer_comb2}")

    gcov_cl_1 = tjp1.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                     tracer_comb2=tracer_comb2,
                                     ccl_tracers=ccl_tracers,
                                     tracer_Noise=tracer_Noise,
                                     two_point_data=tjp0.cl_data,
                                     do_xi=False)
    np.testing.assert_allclose(gcov_cl_1['final_b'],
                               ref_cov0cl[:24, :24])


def depickling(name):
    """get the target data 
    """
    try:
        with open(f"tests/val_{name}.pkl", 'rb') as ff:
            return pickle.load(ff)
    except:
        print(f"missing {name}")
        print(os.listdir("tests/"))

#INPUT
# CCL and sacc input:
with open("tests/data/cosmos_desy1_v2p2p0.pkl", 'rb') as ff:
    cosmo = pickle.load(ff)

cosmo_filename = "tests/data/cosmo_desy1.yaml"

xi_fn = "examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
cl_fn = "examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits"

# Reference Output extracted from ipynb:
with open("tests/data/tjpcov_cl.pkl", "rb") as ff:
    ref_cov0cl = pickle.load(ff)
with open("tests/data/tjpcov_xi.pkl", "rb") as ff:
    ref_cov0xi = pickle.load(ff)

ref_covnobin = depickling("covnobin")  # ell datavectors bins #EDGES
ref_md_ell_bins = depickling('metadata_ell_bins')
ref_thb = depickling('thb')
ref_WT = depickling('WT')  # Wigner Transform object
ref_md_th = depickling('metadata_th')
ref_md_th_bins = depickling('metadata_th_bins')  # th datavectors bins #EDGES
ref_md_ell_bins = depickling('metadata_ell_bins')

# SETUP
tjp0 = cv.CovarianceCalculator.from_yaml(tjpcov_cfg=
                                         "tests/data/conf_tjpcov_minimal.yaml")

ccl_tracers, tracer_Noise = tjp0.get_tracer_info(tjp0.cl_data)


def test_theta():
    """
    Check correct conversion of units in theta

    """
    th_min = 2.5/60
    th_max = 250/60
    ref_theta_edges = np.logspace(np.log10(th_min), np.log10(th_max), 20+1)

    np.testing.assert_array_equal(
        tjp0.theta_edges, ref_theta_edges), 'error theta'


def test_ell():
    """Ensure the correct setup for C_ell
    """
    np.testing.assert_array_equal(tjp0.ell_edges,
                                  ref_md_ell_bins)
    
print(type(tjp0))

def test_set_cosmo():
    tjp0 = cv.CovarianceCalculator(**input_dictionary)
    assert isinstance(tjp0, cv.CovarianceCalculator)

# @pytest.mark.slow
def test_xi_block():
    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')

    print(f"Checking covariance block. \
          Tracer combination {tracer_comb1} {tracer_comb2}")

    gcov_xi_1 = tjp1_cs.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                        tracer_comb2=tracer_comb2,
                                        ccl_tracers=ccl_tracers,
                                        tracer_Noise=tracer_Noise,
                                        two_point_data=tjp0.cl_data,
                                        do_xi=True)
    np.testing.assert_allclose(gcov_xi_1['final'], ref_covnobin)
    return


# @pytest.mark.slow
def test_cl_block():

    tracer_comb1 = ('lens0', 'lens0')
    tracer_comb2 = ('lens0', 'lens0')

    print(f"Checking covariance block. \
          Tracer combination {tracer_comb1} {tracer_comb2}")

    gcov_cl_1 = tjp0.cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                     tracer_comb2=tracer_comb2,
                                     ccl_tracers=ccl_tracers,
                                     tracer_Noise=tracer_Noise,
                                     two_point_data=tjp0.cl_data,
                                     do_xi=False)
    np.testing.assert_allclose(gcov_cl_1['final_b'],
                               ref_cov0cl[:24, :24])


# @pytest.mark.slow
def test_cl_cov():
    print("Comparing Cl covariance (840 data points)")
    gcov_cl = tjp0.get_all_cov()
    np.testing.assert_allclose(gcov_cl,
                               ref_cov0cl[:, :])


@pytest.mark.slow
def test_xi_cov():
    print("Comparing xi covariance (700 data points)")
    covall_xi = tjp1_cs.get_all_cov(do_xi=True)
    np.testing.assert_allclose(covall_xi,
                               ref_cov0xi[:, :])



if __name__ == '__main__':
    import pdb 
    pdb.set_trace()
    test_xi_cov()
    pass

