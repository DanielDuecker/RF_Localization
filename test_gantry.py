import gantry_control
import rf_tools
import numpy as np


dx = 100  # [mm]
dy = 100  # [mm]
"""
meas_time_set = [5]
for meas_time in meas_time_set:

    wp_filename = 'wp_list_inner_space_' + str(dx) +'mm_'+ str(meas_time)+'s.txt'
    rf_tools.wp_generator(wp_filename, [1150, 550], [2250, 1200], [dx, dy], meas_time, False)  # x-axis along belt-drive
"""
gt = gantry_control.GantryControl([0, 3100, 0, 1600], True)



freqtx = [434.1e6, 434.15e6, 434.4e6, 434.45e6]


#gt.start_CalEar(freqtx)
numtx = len(freqtx)
tx_abs_pos = [[1060, 470],  # 433.9MHz
              [1060, 1260],  # 434.1MHz
              [2340, 470],  # 434.3MHz
              [2340, 1260]]  # 434.5MHz

"""
    date = '2017_02_17'
    meas_filename = 'measdata_' + date + '_inner_space_'+str(dx)+'mm_' + str(meas_time) + 's_a.txt'
    gt.process_measurement_sequence(wp_filename, meas_filename, numtx, tx_abs_pos, freqtx)
"""

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
for i in range(3):
    gt.follow_wp(start_wp_outer, wp_list_outer)
"""
run = [1,2,3]
sample_size_set = [32, 256]
for sample_size in sample_size_set:
    for num in run:
        filename = 'rectangle_2017_02_17_inner_symmetric_sample'+str(sample_size)+'_'+str(num)+'_a.txt'
        gt.follow_wp_and_take_measurements(start_wp_inner, wp_list_inner, filename,sample_size)
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
