import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path
import time as t

t.time()

wp_filename_rel_path = path.relpath('Aktuell/wp_list_2018_08_14_grid_meas_d505050.txt')

x0 = [100, 500, 0]
xn = [3000, 1150, 600]
dxdyda = [50, 50, 50]

sdr_type = 'AirSpy'  # 'AirSpy' / 'NooElec'

rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 2, True)

# rf_tools.analyze_measdata_from_file('lin')

Rf = rf.RfEar(sdr_type, 434.0e6, 1e5)

# freq6tx = [433.975e6, 434.52e6, 434.61e6, 434.12e6, 434.275e6, 434.42e6]
plt.ion()

freq6tx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]

tx_6pos = [[830, 430, 600],
           [1854, 435, 600],
           [2874, 445, 600],
           [2884, 1230, 600],
           [1849, 1235, 600],
           [834, 1225, 600]]
Rf.set_txparams(freq6tx, tx_6pos)

Rf.set_samplesize(32)

Rf.plot_power_spectrum_density()
# Rf.plot_txrss_live()



"""

#Rf.get_performance()
#cal.plot_txrss_live()

"""

