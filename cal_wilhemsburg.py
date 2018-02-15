import rf
import rf_tools

Rf = rf.RfEar(434.2e6, 8e4)

freq6tx = [434.00e6, 434.1e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

tx_6pos = [[0, 0],
           [1000, 0],
           [2000, 0],
           [2000, 900],
           [1000, 900],
           [0, 900]]
Rf.set_txparams(freq6tx, tx_6pos)

points14 = [[-500, 300], [0, 300], [500, 300], [1000, 300], [1500, 300], [2000, 300], [2500, 300],
            [-500, 600], [0, 600], [500, 600], [1000, 600], [1500, 600], [2000, 600], [2500, 600]]


measurement_data_filename = 'meas_data_wburg.txt'

#Rf.manual_calibration_for_6_tx(measdata_filename=measurement_data_filename, point_list=points14)
rf_tools.onboard_cal_param(tx_6pos, measdata_filename=measurement_data_filename, param_filename='cal_param.txt')

(alpha_cal, gamma_cal) = rf_tools.get_cal_param_from_file(param_filename='cal_param.txt')
