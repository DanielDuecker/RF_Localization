import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

wp_filename = 'Messung6Antennen.txt'
#wp_filename = 'wp_list_test_2018_07_06_72meas3.txt'
x0 = [2319, 979, 0]
xn = [2319, 979, (2*np.pi)]
dxdyda = [0, 0, (2*np.pi*1/72)]
rf_tools.wp_generator(wp_filename, x0, xn, dxdyda, 3, True)

# rf_tools.analyze_measdata_from_file('lin')

Rf = rf.RfEar(434.2e6, 8e4)

freq6tx = [434.00e6, 434.1e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

tx_6pos = [[520, 430, 0],
           [1540, 430, 0],
           [2570, 430, 0],
           [2570, 1230, 0],
           [1540, 1230, 0],
           [520, 1230, 0]]
Rf.set_txparams(freq6tx, tx_6pos)

# Rf.set_samplesize(32)

# Rf.plot_power_spectrum_density()
Rf.plot_txrss_live()



"""

#Rf.get_performance()
#cal.plot_txrss_live()

"""

