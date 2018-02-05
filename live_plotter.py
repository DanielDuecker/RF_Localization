import socket_server
import estimator_plot_tools
import estimator
import numpy as np
tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]

ekf_plotter = estimator_plot_tools.EKF_Plot(tx_6pos, True)

wp_list = np.array([[1200, 500], [600, 900]])
wprad_list = [800, 900]
ekf_plotter.plot_way_points(wp_list, wprad_list)

"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""

tx_alpha = [0.01149025464796399, 0.016245419273983631, 0.011352095690562954, 0.012125937076390217, 0.0092717529591962722, 0.01295918160582895]
tx_gamma = [-8.5240925102693872, -11.670560994925006, -8.7169295956676116, -8.684528288347666, -5.1895194577206665, -9.8124742816198918]

"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""


server_ip = '192.168.1.23'
server_port = 50008
#soc_server = socket_server.SocServer(server_ip, server_port)
import numpy as np
EKF = estimator.ExtendedKalmanFilter()
while True:
    #rec_data_list = soc_server.soc_send_data_request()
    #if rec_data_list is False:
    #    soc_server.soc_close_restart_connection()
    #    continue  # restart connection and give it a new try

    #msg_time = rec_data_list[0]
    #msg_k = rec_data_list[1]
    #msg_x_est = rec_data_list[2]
    #msg_z_meas = rec_data_list[3]
    #msg_y_est = rec_data_list[4]

    EKF.ekf_prediction()
    EKF.ekf_update()

    msg_x_est = EKF.get_x_est()
    msg_y_est = EKF.get_y_est()
    msg_z_meas = EKF.get_z_meas()

    ekf_plotter.update_meas_circles(msg_z_meas, tx_alpha, tx_gamma, True, msg_y_est)
    ekf_plotter.plot_ekf_pos_live(msg_x_est[0], msg_x_est[1], 100)

