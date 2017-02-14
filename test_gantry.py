import gantry_control
import rf_tools
import numpy as np


dx = 50  # [mm]
dy = 50  # [mm]
#rf_tools.wp_generator('wp_list_2017_02_10_50mm.txt', [0, 0], [3000, 1550], [3000/dx+1, 1550/dy+1], 10.0, True)  # x-axis along belt-drive
gt = gantry_control.GantryControl()

# freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]
freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
#gt.process_measurement_sequence('wp_list_2017_02_10_50mm.txt', 'measdata_2017_02_11_50mm.txt')
gt.follow_wp_and_take_measurements()