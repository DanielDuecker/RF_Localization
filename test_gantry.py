import gantry_control
import rf_tools
import numpy as np


dx = 25  # [mm]
dy = 25  # [mm]
rf_tools.wp_generator('wp_list_tx1_25mm.txt', [750, 250], [1250, 750], [dx, dy], 10.0, True)  # x-axis along belt-drive
gt = gantry_control.GantryControl()

freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
numtx = len(freqtx)
tx_abs_pos = [[1060, 470],  # 433.9MHz
              [1060, 1260],  # 434.1MHz
              [2340, 470],  # 434.3MHz
              [2340, 1260]]  # 434.5MHz

gt.process_measurement_sequence('wp_list_tx1_25mm.txt', 'measdata_2017_02_15_tx1_25mm.txt', numtx, tx_abs_pos, freqtx)

#gt.follow_wp_and_take_measurements()