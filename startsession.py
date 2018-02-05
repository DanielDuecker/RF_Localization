import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


wp_filename = 'wp_list_6tx_innerfield_left_180_half_100mm_2018.txt'
x0 = [80, 500]
xn = [1500, 1200]
dxdy = [100, 100]
#rf_tools.wp_generator(wp_filename, x0, xn, dxdy, 3, True)



#rf_tools.analyze_measdata_from_file('lin')



Rf = rf.RfEar(434.2e6, 8e4)



freq6tx = [434.00e6, 434.1e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]
Rf.set_txparams(freq6tx, tx_6pos)

tx_alpha = [0.01149, 0.01624, 0.01135, 0.01212, 0.00927, 0.012959]
tx_gamma = [-8.52409, -11.6705, -8.7169, -8.684, -5.1895, -9.81247]
#Rf.set_calparams(tx_alpha, tx_gamma)

Rf.set_samplesize(32)

#Rf.map_path_ekf([600,600], 'h_rss')
#
#Rf.plot_power_spectrum_density()
Rf.plot_txrss_live()



"""

#Rf.get_performance()
#cal.plot_txrss_live()

"""

