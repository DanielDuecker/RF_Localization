import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab



#rf_tools.wp_generator([0, 0], [2900, 1500], [59, 31], 10.0, True) # x-axis is along belt-drive



txpos_offset = np.array([1040, 450])

txpos = [[0, 0],  # 433.9MHz
         [0, 780],  # 434.1MHz
         [1270, 0],  # 434.3MHz
         [1270, 780]]  # 434.5MHz


analyze_tx = [1,2,3,4]
#rf_tools.analyse_measdata_from_file(analyze_tx, txpos, txpos_offset, freqtx)


freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]
cal = rf.CalEar(freqtx)

cal.plot_psd()
#cal.get_performance()
#cal.plot_txrss_live()


freqspan = 2e4
freqcenter = np.mean(freqtx)


#loc = rf.LocEar(freqtx, freqspan, alpha, xi, txpos+txpos_offset)

#loc.plot_txdist_live()

x0 = np.array([300, 400])  # initial estimate

#loc.map_path_ekf(x0, True, False, False)
