import gantry_control
import rf_tools
import numpy as np


gt = gantry_control.GantryControl()

rf_tools.wp_generator('wp_list_2017_01_12.txt', [0, 0], [2900, 1500], [59, 31], 10.0, True) # x-axis is along belt-drive

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
gt.process_measurement_sequence('wp_list_2017_01_12.txt', 'measdata_2017_01_12.txt')

