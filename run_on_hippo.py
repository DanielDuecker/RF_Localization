import estimator
import socket_server
import time


# def loop_on_hippo(b_soc_transmission = True, b_client_connected = True,server_ip = '192.168.1.1',server_port = 50007):
b_soc_transmission = True
server_ip = '192.168.1.1'
server_port = 50007

if b_soc_transmission:
    soc_server = socket_server.SocServer(server_ip, server_port)
    # print('wait for 5 seconds')
    # time.sleep(5)

EKF = estimator.ExtendedKalmanFilter()
t0 = time.time()
k = 0
while True:
    runtime = time.time() - t0
    k = k + 1
    EKF.ekf_prediction()
    EKF.ekf_update()

    """ Data Transmission to base station"""

    data = 'time' + str(runtime) + ', k' + str(k) + ' x_est' + str(EKF.get_x_est()) + ' z_meas' + str(EKF.get_z_meas())
    soc_server.soc_process_request(data)

