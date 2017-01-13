#import rf
import rf_tools
import numpy as np


#rf_tools.wp_generator([0, 0], [2900, 1500], [59, 31], 10.0, True) # x-axis is along belt-drive

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

txpos_offset = np.array([1060, 385])
txpos = [[0, 0],  # 433.9MHz
         [0, 800],  # 434.1MHz
         [1270, 50],  # 434.3MHz
         [1270, 750]]  # 434.5MHz
analyze_tx = [1,2,3,4]
rf_tools.analyse_measdata_from_file(analyze_tx, txpos, txpos_offset, freqtx)



"""
# calibration data from measdata_2017_01_09
tx #1 alpha= 0.0134497413559 xi= -10.8274256871
tx #2 alpha= 0.0129485499082 xi= -8.05984834452
tx #3 alpha= 0.0137674277788 xi= -9.1403814896
tx #4 alpha= 0.012530150337 xi= -8.06862854932
"""
#alpha = [0.0134497413559, 0.0129485499082, 0.0137674277788, 0.012530150337]
#xi = [-10.8274256871, -8.05984834452, -9.1403814896, -8.06862854932]


"""
# calibration data from measdata_2017_01_10
#tx #1 alpha= 0.0133680937013 xi= -10.9915388063
#tx #2 alpha= 0.0129727376314 xi= -8.40605996688
#tx #3 alpha= 0.0139948951361 xi= -9.67729971289
#tx #4 alpha= 0.0124357921976 xi= -8.34912917588
alpha = [0.0133680937013, 0.0129727376314, 0.0139948951361, 0.0124357921976]
xi = [-10.9915388063, -8.40605996688, -9.67729971289, -8.34912917588]
"""

"""
# calibration data from measdata_2017_01_11
tx #1 alpha= 0.0133069586558 xi= -11.0365481144
tx #2 alpha= 0.0128027023077 xi= -8.41650949358
tx #3 alpha= 0.0139322778211 xi= -9.65681472185
tx #4 alpha= 0.0124073541579 xi= -8.32949986179

Vectors for convenient copy/paste
alpha = [0.013306958655773716, 0.012802702307698088, 0.013932277821081489, 0.012407354157857986]
xi = [-11.036548114407907, -8.4165094935799942, -9.656814721853733, -8.3294998617936358]
"""


"""
# calibration data from measdata_2017_01_12
tx #1 alpha= 0.01354439012 xi= -11.5191298861
tx #2 alpha= 0.0127396113931 xi= -8.65152096901
tx #3 alpha= 0.0128867071391 xi= -9.26528070888
tx #4 alpha= 0.0127889287934 xi= -8.76417416152

Vectors for convenient copy/paste
alpha = [0.013544390120028761, 0.012739611393085375, 0.012886707139092372, 0.012788928793384974]
xi = [-11.519129886119973, -8.6515209690122905, -9.2652807088786631, -8.7641741615232593]

"""
"""
# calibration data from measdata_2017_01_12 - new antenna #3 build by Rene on 2017/01/11
tx #1 alpha= 0.0129627275519 xi= -11.6376282983
tx #2 alpha= 0.0128643798655 xi= -8.17647776207
tx #3 alpha= 0.0128420400656 xi= -9.5934783667
tx #4 alpha= 0.012219417791 xi= -7.973071723

Vectors for convenient copy/paste
alpha = [0.012962727551932132, 0.012864379865473508, 0.012842040065627317, 0.012219417791040854]
xi = [-11.637628298341465, -8.1764777620721336, -9.5934783667043018, -7.9730717230049883]
"""
#cal = rf.CalEar(freqtx)

#cal.plot_psd()
#cal.get_performance()
#cal.plot_txrss_live()


freqspan = 2e4
freqcenter = np.mean(freqtx)


#loc = rf.LocEar(freqtx, freqspan, alpha, xi, txpos)
#loc.plot_txdist_live()

x0 = np.array([300, 400])  # initial estimate

#loc.map_path_ekf(x0, False, False, True)
