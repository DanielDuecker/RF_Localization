#import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

"""
delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)


print('xshape' + str(X.shape))
print('xshape' + str(Z.shape))
# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')

plt.show()

"""

#rf_tools.wp_generator([0, 0], [2900, 1500], [59, 31], 10.0, True) # x-axis is along belt-drive

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]


#txpos_offset = np.array([1060, 450])
txpos_offset = np.array([1040, 450])


txpos = [[0, 0],  # 433.9MHz
         [0, 800],  # 434.1MHz
         [1270, 50],  # 434.3MHz
         [1270, 750]]  # 434.5MHz


"""
# turned by 180deg
txpos = [[1270, 750],  # 433.9MHz
         [1270, -50],  # 434.1MHz
         [0, 700],  # 434.3MHz
         [0, 0]]  # 434.5MHz
"""
analyze_tx = [1,2,3,4]
#rf_tools.analyse_measdata_from_file(analyze_tx, txpos, txpos_offset, freqtx)



"""
# calibration data from measdata_2017_01_09
tx #1 alpha= 0.0134497413559 gamma= -10.8274256871
tx #2 alpha= 0.0129485499082 gamma= -8.05984834452
tx #3 alpha= 0.0137674277788 gamma= -9.1403814896
tx #4 alpha= 0.012530150337 gamma= -8.06862854932
"""
#alpha = [0.0134497413559, 0.0129485499082, 0.0137674277788, 0.012530150337]
#gamma = [-10.8274256871, -8.05984834452, -9.1403814896, -8.06862854932]


"""
# calibration data from measdata_2017_01_10
Number of gridpoints: 1829
#tx #1 alpha= 0.0133680937013 gamma= -10.9915388063
#tx #2 alpha= 0.0129727376314 gamma= -8.40605996688
#tx #3 alpha= 0.0139948951361 gamma= -9.67729971289
#tx #4 alpha= 0.0124357921976 gamma= -8.34912917588
alpha = [0.0133680937013, 0.0129727376314, 0.0139948951361, 0.0124357921976]
gamma = [-10.9915388063, -8.40605996688, -9.67729971289, -8.34912917588]
"""

"""
# calibration data from measdata_2017_01_11
Number of gridpoints: 1829
tx #1 alpha= 0.0133069586558 gamma= -11.0365481144
tx #2 alpha= 0.0128027023077 gamma= -8.41650949358
tx #3 alpha= 0.0139322778211 gamma= -9.65681472185
tx #4 alpha= 0.0124073541579 gamma= -8.32949986179

Vectors for convenient copy/paste
alpha = [0.013306958655773716, 0.012802702307698088, 0.013932277821081489, 0.012407354157857986]
gamma = [-11.036548114407907, -8.4165094935799942, -9.656814721853733, -8.3294998617936358]
"""


"""
# calibration data from measdata_2017_01_12
Number of gridpoints: 1829
tx #1 alpha= 0.01354439012 gamma= -11.5191298861
tx #2 alpha= 0.0127396113931 gamma= -8.65152096901
tx #3 alpha= 0.0128867071391 gamma= -9.26528070888
tx #4 alpha= 0.0127889287934 gamma= -8.76417416152

Vectors for convenient copy/paste
alpha = [0.013544390120028761, 0.012739611393085375, 0.012886707139092372, 0.012788928793384974]
gamma = [-11.519129886119973, -8.6515209690122905, -9.2652807088786631, -8.7641741615232593]

"""
"""
# calibration data from measdata_2017_01_13 - new antenna #3 build by Rene on 2017/01/11
Number of gridpoints: 1829
tx #1 alpha= 0.0129627275519 gamma= -11.6376282983
tx #2 alpha= 0.0128643798655 gamma= -8.17647776207
tx #3 alpha= 0.0128420400656 gamma= -9.5934783667
tx #4 alpha= 0.012219417791 gamma= -7.973071723

Vectors for convenient copy/paste
alpha = [0.012962727551932132, 0.012864379865473508, 0.012842040065627317, 0.012219417791040854]
gamma = [-11.637628298341465, -8.1764777620721336, -9.5934783667043018, -7.9730717230049883]
"""

"""
# calibration data from measdata_2017_01_14 - new antenna #3 build by Rene on 2017/01/11
Number of gridpoints: 1829
tx #1 alpha= 0.0125232795169 gamma= -11.0548977228
tx #2 alpha= 0.0126372069151 gamma= -7.90889418955
tx #3 alpha= 0.0121991534121 gamma= -8.91987532197
tx #4 alpha= 0.0122998581165 gamma= -8.08970877326

Vectors for convenient copy/paste
alpha = [0.012523279516922432, 0.01263720691510125, 0.012199153412068222, 0.012299858116518337]
gamma = [-11.054897722752985, -7.9088941895534521, -8.9198753219699842, -8.0897087732576924]
"""

"""
# calibration data from measdata_2017_01_14 - new antenna #3 build by Rene on 2017/01/11
Number of gridpoints: 384
tx #1 alpha= 0.0123814733364 gamma= -10.9197178211
tx #2 alpha= 0.0120204352138 gamma= -7.61056146554
tx #3 alpha= 0.0116849528917 gamma= -8.56105830355
tx #4 alpha= 0.0121945410021 gamma= -8.16667820173

Vectors for convenient copy/paste
alpha = [0.012381473336446973, 0.012020435213791609, 0.011684952891692585, 0.012194541002146064]
gamma = [-10.919717821111632, -7.6105614655357741, -8.5610583035501779, -8.1666782017252384]
"""

#cal = rf.CalEar(freqtx)

#cal.plot_psd()
#cal.get_performance()
#cal.plot_txrss_live()


freqspan = 2e4
freqcenter = np.mean(freqtx)


#loc = rf.LocEar(freqtx, freqspan, alpha, gamma, txpos)
#loc.plot_txdist_live()

x0 = np.array([300, 400])  # initial estimate

#loc.map_path_ekf(x0, False, False, True)
