import os
import pytest
import numpy as np
import sacc
import pickle
import pyccl as ccl
import pymaster as nmt
from tjpcov_new.covariance_builder import CovarianceFourier


root = "./tests/benchmarks/32_DES_tjpcov_bm/"
outdir = root + 'tjpcov_tmp/'
input_yml = os.path.join(root, "tjpcov_conf_minimal.yaml")
input_sacc = sacc.Sacc.load_fits(root + 'cls_cov.fits')

# Create temporal folder
os.makedirs('tests/tmp/', exist_ok=True)


class CovarianceFourierTester(CovarianceFourier):
    # Based on https://stackoverflow.com/a/28299369
    def _build_matrix_from_blocks(self, blocks, tracers_cov):
        super()._build_matrix_from_blocks(blocks, tracers_cov)

    def get_covariance_block(self, **kwargs):
        super().get_covariance_block(**kwargs)


def get_dummy_sacc():
    s = sacc.Sacc()
    s.add_tracer('map', 'PLAcv', quantity='cmb_convergence', spin=0,
                 ell=None, beam=None)
    s.add_tracer('NZ', 'DESgc__0', quantity='galaxy_density', spin=0,
                 nz=None, z=None)
    s.add_tracer('NZ', 'DESwl__0', quantity='galaxy_shear', spin=2,
                 nz=None, z=None)
    s.add_tracer('misc', 'ForError', quantity='generic')

    return s


def get_nmt_bin(lmax=95):
    bpw_edges = np.array([0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                           78, 84, 90, 96])
    if lmax != 95:
        # lmax + 1 because the upper edge is not included
        bpw_edges = bpw_edges[bpw_edges < lmax+1]
        bpw_edges[-1] = lmax+1

    return  nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])


def test_build_matrix_from_blocks():
    pass


def test_get_datatypes_from_ncell():
    cb = CovarianceFourierTester(input_yml)

    with pytest.raises(ValueError):
        cb.get_datatypes_from_ncell(0)

    with pytest.raises(ValueError):
        cb.get_datatypes_from_ncell(3)

    assert cb.get_datatypes_from_ncell(1) == ['cl_00']
    assert cb.get_datatypes_from_ncell(2) == ['cl_0e', 'cl_0b']
    assert cb.get_datatypes_from_ncell(4) == ['cl_ee', 'cl_eb', 'cl_be',
                                              'cl_bb']


def test_get_ell_eff():
    cb = CovarianceFourierTester(input_yml)

    # Using this because the data was generated with NaMaster. We could read
    # the sacc file but then we would be doing the same as in the code.
    bins = get_nmt_bin()
    ells = bins.get_effective_ells()

    assert np.all(cb.get_ell_eff() == ells)


def test_get_tracer_comb_ncell():
    cb = CovarianceFourierTester(input_yml)

    # Use dummy file to test for cmb_convergence too
    cb.io.sacc_file = get_dummy_sacc()

    assert cb.get_tracer_comb_ncell(('PLAcv', 'PLAcv')) == 1
    assert cb.get_tracer_comb_ncell(('PLAcv', 'DESgc__0')) == 1
    assert cb.get_tracer_comb_ncell(('DESgc__0', 'DESgc__0')) == 1
    assert cb.get_tracer_comb_ncell(('PLAcv', 'DESwl__0')) == 2
    assert cb.get_tracer_comb_ncell(('DESgc__0', 'DESwl__0')) == 2
    assert cb.get_tracer_comb_ncell(('DESwl__0', 'DESwl__0')) == 4
    assert cb.get_tracer_comb_ncell(('DESwl__0', 'DESwl__0'),
                                    independent=True) == 3


def test_get_tracer_info():
    cb = CovarianceFourierTester(input_yml)
    ccl_tracers1, tracer_noise1 = cb.get_tracer_info()
    ccl_tracers, tracer_noise, tracer_noise_coupled = \
        cb.get_tracer_info(return_noise_coupled=True)

    # Check that when returnig the coupled noise, the previous output is the
    # same
    assert ccl_tracers is ccl_tracers1
    assert tracer_noise is tracer_noise1
    assert tracer_noise_coupled is None

    # Check noise from formula
    arc_min = 1/60 * np.pi / 180  # arc_min in radians
    Ngal = 26 / arc_min**2  # Number galaxy density
    sigma_e = 0.26

    for tr, nl in tracer_noise.items():
        if 'gc' in tr:
            assert np.abs(nl / (1/Ngal) - 1) < 1e-5
        else:
            assert np.abs(nl / (sigma_e**2/Ngal) - 1) < 1e-5

    # TODO: We should check the CCL tracers are the same
    for tr, ccltr in ccl_tracers.items():
        if 'gc' in tr:
            assert isinstance(ccltr, ccl.NumberCountsTracer)
        elif 'wl' in tr:
            assert isinstance(ccltr, ccl.WeakLensingTracer)
        elif 'cv' in tr:
            assert isinstance(ccltr, ccl.CMBLensingTracer)

    # Check tracer_noise_coupled. Modify the sacc file to add metadata
    # information for the tracer noise
    cb = CovarianceFourierTester(input_yml)
    cb.io.get_sacc_file()  # To have the sacc file stored it in cb.io.sacc_file
    coupled_noise = {}
    for i, tr in enumerate(cb.io.sacc_file.tracers.keys()):
        coupled_noise[tr] = i
        cb.io.sacc_file.tracers[tr].metadata['n_ell_coupled'] = i

    ccl_tracers, tracer_noise, tracer_noise_coupled = \
        cb.get_tracer_info(return_noise_coupled=True)

    for tr, nl in tracer_noise_coupled.items():
        assert coupled_noise[tr] == nl

    # Check that tracer_noise_coupled will be None if one of them is missing
    cb = CovarianceFourierTester(input_yml)

    ccl_tracers, tracer_noise, tracer_noise_coupled = \
        cb.get_tracer_info(return_noise_coupled=True)

    assert tracer_noise_coupled is None
