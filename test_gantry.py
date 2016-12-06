import gantry_control
import numpy as np


gt = gantry_control.GantryControl()

# gantry_control.wp_generator('wp_list.txt')

gt.start_CalEar(freqtx=[433.9e6, 434e6])
gt.process_measurement_sequence('wp_list.txt', 'measdata2.txt')





# manually tuned parameter
alpha = [0.12615852725, 0.117592701848, 0.114243243761]
xi = [10.6176901836, 12.9628114874, 23.2984322235]
freqtx = [433.91e6, 434.16e6, 433.7e6]
# absolute tx position
txpos = np.array([[0.0, 0.0],     # 433,91MHz
                  [80.0, 0.0],    # 434,16MHz
                  [40.0, 62.0]])  # 433,70MHz

# gt.start_LocEar(alpha, xi, txpos, freqtx)

# gt.process_measurement_sequence('wp_list.txt', 'measdata.txt')
