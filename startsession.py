import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

wp_filename = 'wp_list_2018_07_31_grid_meas_0deg_d20_2sec_no1.txt'

x0 = [400, 460, 0]
xn = [3000, 1200, 0]
dxdyda = [20, 20, 0]
rf_tools.wp_generator(wp_filename, x0, xn, dxdyda, 2, True)

# rf_tools.analyze_measdata_from_file('lin')

Rf = rf.RfEar(434.2e6, 8e4)

#freq6tx = [433.975e6, 434.52e6, 434.61e6, 434.12e6, 434.275e6, 434.42e6]

freq6tx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]
tx_6pos = [[770, 432, 0],
           [1794, 437, 0],
           [2814, 447, 0],
           [2824, 1232, 0],
           [1789, 1237, 0],
           [774, 1227, 0]]
Rf.set_txparams(freq6tx, tx_6pos)

# Rf.set_samplesize(32)

#Rf.plot_power_spectrum_density()
# Rf.plot_txrss_live()



"""

#Rf.get_performance()
#cal.plot_txrss_live()

"""

