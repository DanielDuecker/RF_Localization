import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path

<<<<<<< HEAD
wp_filename_rel_path = path.relpath('Aktuell/wp_list_2018_08_02_grid_meas_0deg_d50_2sec_no1.txt')
=======
wp_filename_rel_path = path.relpath('Aktuell/wp_list_2018_08_07_grid_meas_0deg_d50_2sec_no1.txt')
>>>>>>> 7aeddf90b11b28fb02d6f1b0ff9d1c536a9bdba4

x0 = [600, 500, 0]
xn = [3000, 1150, 0]
dxdyda = [50, 50, 0]
<<<<<<< HEAD
rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)
=======
>>>>>>> 7aeddf90b11b28fb02d6f1b0ff9d1c536a9bdba4

sdr_type = 'NooElec'  # 'AirSpy' / 'NooElec'

rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)

# rf_tools.analyze_measdata_from_file('lin')

Rf = rf.RfEar(sdr_type, 434.2e6, 1e5)

# freq6tx = [433.975e6, 434.52e6, 434.61e6, 434.12e6, 434.275e6, 434.42e6]

freq6tx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]
tx_6pos = [[770, 432, 0],
           [1794, 437, 0],
           [2814, 447, 0],
           [2824, 1232, 0],
           [1789, 1237, 0],
           [774, 1227, 0]]
# Rf.set_txparams(freq6tx, tx_6pos)

# Rf.set_samplesize(32)

# Rf.plot_power_spectrum_density()
# Rf.plot_txrss_live()



"""

#Rf.get_performance()
#cal.plot_txrss_live()

"""

