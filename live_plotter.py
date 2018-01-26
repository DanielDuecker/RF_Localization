import socket_server
import estimator_plot_tools

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]

ekf_plotter = estimator_plot_tools.EKF_Plot(tx_6pos, True)
soc_client = socket_server.SocClient('192.168.1.1', 50008)

tx_alpha = [0.01149025464796399, 0.016245419273983631, 0.011352095690562954, 0.012125937076390217, 0.0092717529591962722, 0.01295918160582895]
tx_gamma = [-8.5240925102693872, -11.670560994925006, -8.7169295956676116, -8.684528288347666, -5.1895194577206665, -9.8124742816198918]

"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""
while True:
    rec_data_list = soc_client.soc_send_data_request()

    msg_time = rec_data_list[0]
    msg_k = rec_data_list[1]
    msg_x_est = rec_data_list[2]
    msg_z_meas = rec_data_list[3]
    msg_y_est = rec_data_list[4]
    #print(msg_x_est)
    ekf_plotter.add_data_to_plot(msg_x_est, 'bo')
    ekf_plotter.plot_meas_circles(msg_z_meas, msg_y_est, tx_alpha, tx_gamma)
    ekf_plotter.update_plot()


    print('k = ' + str(msg_k) + ', x_est  = ' + str([msg_x_est[0], msg_x_est[1]]))
    #print(', x_est  = ' + str([msg_x_est[0], msg_x_est[1]]), 'y_est = ' + str(msg_y_est))

    # ekf_plotter.add_data_to_plot_list(msg_x_est[0], msg_x_est[1])
    # ekf_plotter.update_live(10)
