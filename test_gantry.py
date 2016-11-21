import gantry_control


gt = gantry_control.GantryControl()

#gantry_control.wp_generator('wp_list.txt')
gt.start_sdr()
gt.process_measurement_sequence('wp_list.txt', 'measdata.txt')
