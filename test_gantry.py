import gantry_control
import rf_tools
import numpy as np


dx = 50  # [mm]
dy = 50  # [mm]
rf_tools.wp_generator('wp_list_inner_space_50mm_5s.txt', [1150, 550], [2250, 1200], [dx, dy], 5.0, True)  # x-axis along belt-drive

gt = gantry_control.GantryControl([0, 3100, 0, 1600], True)



freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]

#gt.follow_wp_trajectory(400, 2000, 50)

gt.start_CalEar(freqtx)
numtx = len(freqtx)
tx_abs_pos = [[1060, 470],  # 433.9MHz
              [1060, 1260],  # 434.1MHz
              [2340, 470],  # 434.3MHz
              [2340, 1260]]  # 434.5MHz



gt.process_measurement_sequence('wp_list_inner_space_50mm_5s.txt', 'measdata_2017_02_15_inner_space_50mm_5s.txt', numtx, tx_abs_pos, freqtx)


start_wp = [1500, 600]
wp_list = [[2200, 600],
           [2200, 1000],
           [1500, 1000],
           [1500, 600]]

start_wp_inner = [1260,670]
wp_list_inner = [[2140,670],
                 [2140,1060],
                 [1260,1060],
                 [1260,670]]

start_wp_outer = [860,270]
wp_list_outer = [[2540,270],
                 [2540,1460],
                 [860,1460],
                 [860,270]]


run = [1,2,3]
sample_size = 256
"""
for num in run:
    filename = 'rectangle_2017_02_17_outer_symmetric_sample'+str(sample_size)+'_'+str(num)+'_.txt'
    gt.follow_wp_and_take_measurements(start_wp_outer, wp_list_outer, filename,sample_size)
"""



position_list = [[1260,1060],  # inner
                 [1260,865],  # inner
                 [1260,670],  # inner
                 [1700,670],  # inner
                 [2140,670],  # inner
                 [1060,1460],  # outer
                 [1060,875],  # outer
                 [1060,270],  # outer
                 [1800,270],  # outer
                 [2540,270]]  # outer
"""
pos_list = position_list
run = [1,2,3]
sample_size = 256
for meas_point in pos_list:
    for num in run:
        meas_time = 10
        filename = 'pos_hold_2017_02_17_' + str(meas_point[0]) +'_'+ str(meas_point[1]) +'_'+str(meas_time)+ 's_sample' + str(sample_size) + '_' + str(num) + '_.txt'
        gt.position_hold_measurements(meas_point, meas_time,filename,sample_size)
"""
