import numpy as np

class ClusterCovarianceArgs(object):

    nz = 128
    
    z_min = 0.3
    z_max = 1.2
    z_bin_range = 0.05

    min_richness = 10
    max_richness = 100
    richness_bin_range = 30

    min_mass = np.log(1e13)
    max_mass = np.log(1e16)