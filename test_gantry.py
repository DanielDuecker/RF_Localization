import gantry_control
import rf_tools
import numpy as np


dx = 50  # [mm]
dy = 50  # [mm]
rf_tools.wp_generator('wp_list_test.txt', [0, 0], [3000, 1550], [dx, dy], 10.0, True)  # x-axis along belt-drive
gt = gantry_control.GantryControl()

# freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]
freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
numtx = len(freqtx)
tx_abs_pos = [[0, 0],  # 433.9MHz
              [0, 780],  # 434.1MHz
              [1270, 0],  # 434.3MHz
              [1270, 780]]  # 434.5MHz

gt.process_measurement_sequence('wp_list_test.txt', 'measdata_test.txt', numtx, tx_abs_pos, freqtx)

#gt.follow_wp_and_take_measurements()