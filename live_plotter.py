import socket_server
import estimator
import estimator_plot_tools
import numpy as np
import time as t

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [520, 1230]]
ekf_plotter = estimator_plot_tools.EKF_Plot(tx_6pos, bplot_circles=False)
"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""
tx_alpha = [0.01149025464796399, 0.016245419273983631, 0.011352095690562954, 0.012125937076390217, 0.0092717529591962722, 0.01295918160582895]
tx_gamma = [-0.49471304043015696, -1.2482393190627841, -0.17291318936462172, -0.61587988305564456, 0.99831151034040444, 0.85711994311461936]
"""
WARNING!!!!!!
IF YOU CHANGE THE ALPHA + GAMMA VALUES IN EKF YOU HAVE!!!! to change them here MANUALLY!!!
"""
server_ip = '192.168.0.100'  # thinkpad ethernet
#server_ip = '192.168.0.101'  # thinkpad wifi-intern
server_port = 50008

soc_server = socket_server.SocServer(server_ip, server_port)


#EKF = estimator.ExtendedKalmanFilter()
act_time = float(t.time())
while True:
    act_time_1 = float(t.time())
    rec_data_list = soc_server.soc_send_data_request()

    # if rec_data_list is False:
    #    soc_server.soc_close_restart_connection()
    #    continue  # restart connection and give it a new try


    msg_time = rec_data_list[0]
    msg_k = rec_data_list[1]
    msg_x_est_temp = rec_data_list[2]
    msg_x_est = [msg_x_est_temp[0]*1000, msg_x_est_temp[1]*1000]
    #print('x= ' + str(msg_x_est))
    msg_yaw_rad = rec_data_list[3]
    msg_z_meas = rec_data_list[4]
    msg_y_est = rec_data_list[5]
    msg_next_wp_temp = rec_data_list[6]
    msg_next_wp = [msg_next_wp_temp[0]*1000, msg_next_wp_temp[1]*1000]
    #print('wp=' + str(msg_next_wp))

    """
    EKF.ekf_prediction()
    EKF.ekf_update()
    msg_x_est = EKF.get_x_est()
    msg_y_est = EKF.get_y_est()
    msg_z_meas = EKF.get_z_meas()
    msg_yaw_rad = 1.3
    msg_y_est = [-60,-60,-60,-60,-60,-60]
    """

    ekf_plotter.add_x_est_to_plot(msg_x_est, msg_yaw_rad)
    ekf_plotter.update_next_wp(msg_next_wp)
    # ekf_plotter.update_meas_circles(msg_z_meas, tx_alpha, tx_gamma, msg_y_est, b_plot_yest=True, )
    ekf_plotter.plot_ekf_pos_live(b_yaw=True, b_next_wp=True, b_plot_gantry=False, numofplottedsamples=200)

    print('loop_time = ' + str(t.time() - act_time_1))
