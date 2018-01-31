import socket_server
import estimator_plot_tools
import estimator

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]

ekf_plotter = estimator_plot_tools.EKF_Plot(tx_6pos, True)

tx_alpha = [0.01149025464796399, 0.016245419273983631, 0.011352095690562954, 0.012125937076390217, 0.0092717529591962722, 0.01295918160582895]
tx_gamma = [-8.5240925102693872, -11.670560994925006, -8.7169295956676116, -8.684528288347666, -5.1895194577206665, -9.8124742816198918]

"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""


server_ip = '192.168.88.128'
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
    #ekf_plotter.add_data_to_plot(msg_x_est, 'bo')
    #ekf_plotter.plot_meas_circles(msg_z_meas, msg_y_est, tx_alpha, tx_gamma)
    #ekf_plotter.update_plot()


    #print('k = ' + str(msg_k) + ', x_est  = ' + str([msg_x_est[0], msg_x_est[1]]))
    #print(', x_est  = ' + str([msg_x_est[0], msg_x_est[1]]), 'y_est = ' + str(msg_y_est))
    #msg_x_est = np.array([1000, 1000])
    wplist = np.array([[1200, 500], [600, 900]])
    radlist = [800, 900]
    ekf_plotter.add_data_to_plot_list(msg_x_est[0], msg_x_est[1])

    ekf_plotter.update_live(100, True, wplist, radlist)
    #ekf_plotter.plot_way_points(wplist, radlist)
