import gantry_control
import rf_tools
import numpy as np


#rf_tools.wp_generator([0, 0], [2900, 1500], [59, 31], 10.0, True)  # x-axis along belt-drive - 1829 points
#rf_tools.wp_generator([0, 0], [2900, 1500], [24, 16], 10.0, True)  # x-axis along belt-drive - 384 points

dx = 25  # [mm]
dy = 25  # [mm]
# rf_tools.wp_generator([0, 0], [2900, 1500], [2900/dx+1, 1500/dy+1], 10.0, True)  # x-axis along belt-drive - 7137 points

gt = gantry_control.GantryControl()

freqtx = [433.9e6, 434.1e6, 434.3e6, 434.5e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
gt.process_measurement_sequence()

