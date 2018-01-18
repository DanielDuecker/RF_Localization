import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import gantry_control as gc

Rf = rf.RfEar(434.2e6, 1e4)

freq_1tx = [434.15e6]
tx_1pos = [0,0]
Rf.set_txparams(freq_1tx, tx_1pos)

tx_alpha = [0.01149]
tx_gamma = [-8.52409]
Rf.set_calparams(tx_alpha, tx_gamma)


#Rf.plot_power_spectrum_density()
#Rf.plot_txrss_live()


Rf.manual_calibration_for_one_tx('testfilename.txt', 1)  # meastime = 5s

def analyze_measdata_from_file_1tx():





