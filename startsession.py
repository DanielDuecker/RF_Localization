import rf
import rf_tools
import numpy as np

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

txpos_offset = np.array([1060, 385])
txpos = [[0, 0],  # 433.9MHz
         [0, 800],  # 434.1MHz
         [1270, 50],  # 434.3MHz
         [1270, 750]]  # 434.5MHz
#rf_tools.analyse_measdata_from_file('measdata_2017_01_09.txt', txpos, txpos_offset, freqtx)



"""
# calibration data from measdata_2017_01_09
tx #1 alpha= 0.0134497413559 xi= -10.8274256871
tx #2 alpha= 0.0129485499082 xi= -8.05984834452
tx #3 alpha= 0.0137674277788 xi= -9.1403814896
tx #4 alpha= 0.012530150337 xi= -8.06862854932
"""
alpha = [0.0134497413559, 0.0129485499082, 0.0137674277788, 0.012530150337]
xi = [-10.8274256871, -8.05984834452, -9.1403814896, -8.06862854932]


"""
# calibration data from measdata_2017_01_10
#tx #1 alpha= 0.0133680937013 xi= -10.9915388063
#tx #2 alpha= 0.0129727376314 xi= -8.40605996688
#tx #3 alpha= 0.0139948951361 xi= -9.67729971289
#tx #4 alpha= 0.0124357921976 xi= -8.34912917588
alpha = [0.0133680937013, 0.0129727376314, 0.0139948951361, 0.0124357921976]
xi = [-10.9915388063, -8.40605996688, -9.67729971289, -8.34912917588]
"""

#cal = rf.CalEar(freqtx)

#cal.plot_psd()
#cal.get_performance()
#cal.plot_txrss_live()


freqspan = 2e4
freqcenter = np.mean(freqtx)


loc = rf.LocEar(freqtx, freqspan, alpha, xi, txpos)
#loc.plot_txdist_live()

x0 = np.array([300, 400])  # initial estimate

loc.map_path_ekf(x0, False, False, True)
