import rf
import rf_tools

Rf = rf.RfEar(434.2e6, 8e4)


freq6tx = [434.00e6, 434.1e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

""" TX position StillWasserBecker """
# tx_6pos = [[520.0, 430.0], [1540.0, 430.0], [2570.0, 430.0], [2570.0, 1230.0], [1540.0, 1230.0], [530.0, 1230.0]]
""" TX position - origin at Beacon #1 """
tx_6pos = tx_6pos = [[0, 0], [1000, 0], [2000, 0], [2000, 900], [1000, 900], [0, 900]]


Rf.set_txparams(freq6tx, tx_6pos)

Rf.set_samplesize(32)

# Rf.plot_power_spectrum_density()
# Rf.plot_txrss_live()


