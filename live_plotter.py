import socket_server
import estimator_plot_tools

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]

ekf_plotter = estimator_plot_tools.EKF_Plot(tx_6pos)
soc_client = socket_server.SocClient('192.168.1.1', 50008)


while True:
    rec_data_list = soc_client.soc_send_data_request()

    msg_time = rec_data_list[0]
    msg_k = rec_data_list[1]
    msg_x_est = rec_data_list[2]
    msg_z_meas = rec_data_list[3]
    #print(msg_x_est)
    ekf_plotter.add_data_to_plot(msg_x_est, 'bo')
    ekf_plotter.update_plot()
    print('k = ' + str(msg_k) + ', x_est  = ' + str([msg_x_est[0], msg_x_est[1]]))

    # ekf_plotter.add_data_to_plot_list(msg_x_est[0], msg_x_est[1])
    # ekf_plotter.update_live(10)
