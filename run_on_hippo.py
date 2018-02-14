import socket_server
import time
import estimator
import numpy as np


host_ip = '192.168.0.100'  # thinkpad ethernet
host_ip = '192.168.0.101'  # thinkpad wifi-intern

host_port = 50008

soc_client = socket_server.SocClient(host_ip, host_port)

EKF = estimator.ExtendedKalmanFilter()
t0 = time.time()
k = 0
while True:
    runtime = time.time() - t0
    k = k + 1
    EKF.ekf_prediction()
    EKF.ekf_update()
    EKF.check_valid_position_estimate()

    EKF.save_to_txt()

    """ Data Transmission to base station """
    yaw_rad = 0.0

    filename = "home/pi/src/mavlink_communication_c_uart/YAW_r_des.txt"
    readdata = []
    with open(filename) as f:
        for line in f:
            readdata.append([float(n) for n in line.strip().split(',')])
        f.close

    print(readdata)

    data_list = [runtime, k, EKF.get_x_est(), EKF.get_z_meas(), EKF.get_y_est(), yaw_rad]
    soc_client.soc_process_server_request(data_list)

    print(k)
