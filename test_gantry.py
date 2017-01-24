import gantry_control
import rf_tools
import numpy as np


dx = 50  # [mm]
dy = 50  # [mm]
rf_tools.wp_generator('wp_list_2017_01_19.txt', [0, 0], [2900, 1500], [2900/dx+1, 1500/dy+1], 10.0, True)  # x-axis along belt-drive
gt = gantry_control.GantryControl()

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
gt.process_measurement_sequence('wp_list_2017_01_19.txt', 'measdata_2017_01_19.txt')
