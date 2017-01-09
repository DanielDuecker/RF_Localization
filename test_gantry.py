import gantry_control
import rf_tools
import numpy as np


gt = gantry_control.GantryControl()

rf_tools.wp_generator('wp_list_2016_12_31.txt', [1200, 200], [2400, 1400], [13, 13], 10.0, True) # x-axis is along belt-drive

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

gt.start_CalEar(freqtx)
gt.process_measurement_sequence('wp_list_2016_12_31.txt', 'measdata_2016_12_31.txt')


