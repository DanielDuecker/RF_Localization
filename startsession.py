import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#"""
wp_filename = 'wp_list_6tx_Fullfield_20mm_2018.txt'
x0 = [0,0]
xn = [3000, 1600]
dxdy = [20, 20]

#rf_tools.wp_generator(wp_filename, x0, xn, dxdy, 5, True)
#"""

txpos_offset = np.array([0, 0])

txpos_tuning = [[0, 0],  # 433.9MHz
                [0, 0],  # 434.1MHz
                [0, 0],  # 434.3MHz
                [0, 0]]  # 434.5MHz

analyze_tx = [1, 2, 3, 4]

#rf_tools.analyse_measdata_from_file(analyze_tx, txpos_tuning)



#Rf = rf.RfEar(434.2e6, 1e4)



freq6tx = [434.00e6, 434.15e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]
#Rf.set_txparams(freq6tx, tx_6pos)
tx_alpha = [0.01149, 0.01624, 0.01135, 0.01212, 0.00927, 0.012959]
tx_gamma = [-8.52409, -11.6705, -8.7169, -8.684, -5.1895, -9.81247]
#Rf.set_calparams(tx_alpha, tx_gamma)

#Rf.set_samplesize(32)

#Rf.map_path_ekf([600,600], 'h_rss')

#Rf.plot_power_spectrum_density()
#Rf.plot_txrss_live()


import socket_server

soc_client = socket_server.SocClient('192.168.1.100', 50007)
while True:
    print(soc_client.soc_send_data_request())




#Rf.get_performance()
#cal.plot_txrss_live()

"""
freqspan = 2e4
freqcenter = np.mean(freqtx)

alpha = [0.013854339628529109, 0.0071309466013866158, 0.018077579531274993, 0.016243668091798915]
gamma = [-1.5898021024559508, 2.0223747861988461, -5.6650866655302714, -5.1158161676293972]
tx_abs_pos = [[1060, 470],  # 433.9MHz
              [1060, 1260],  # 434.1MHz
              [2340, 470],  # 434.3MHz
              [2340, 1260]]  # 434.5MHz

"""

"""
loc = rf.LocEar(434.0e6, freqtx, freqspan, alpha, gamma, tx_abs_pos)

#loc.plot_txdist_live()

x0 = np.array([600, 400])  # initial estimate
loc.set_samplesize(16)
loc.map_path_ekf(x0, 'h_rss', True, False, False)
"""