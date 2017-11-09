import rf as rf

# This file is the minimal running version
# it starts detaches the DVB-T-dongle and starts the power density plot


Rf = rf.RfEar(434.2e6, 1e4)


Rf.plot_power_spectrum_density()
