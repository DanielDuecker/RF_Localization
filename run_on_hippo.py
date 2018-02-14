import socket_server
import time
import estimator
import numpy as np


#host_ip = '192.168.0.100'  # thinkpad ethernet
#host_ip = '192.168.0.101'  # thinkpad wifi-intern

#host_port = 50008

#soc_client = socket_server.SocClient(host_ip, host_port)

EKF = estimator.ExtendedKalmanFilter()
t0 = time.time()
k = 0
while True:
    runtime = time.time()-t0
    t1 = time.time()
    
    k = k + 1
    EKF.ekf_prediction()
    EKF.ekf_update()
    EKF.check_valid_position_estimate()

    EKF.save_to_txt()
    z = EKF.get_z_meas()
    y = EKF.get_y_est()
    
    f_EKF = open("z_y.txt", "w")
    f_EKF.write(
           str(z[0])+ "," +str(z[1])+"," +str(z[2])+","+str(z[3])+","+str(z[4])+","+str(z[5])+ ","+str(y[0]) +","+str(y[1]) +","+str(y[2]) +","+str(y[3]) +","+str(y[4]) +","+str(y[5]))
    f_EKF.close
        
    
    """ Data Transmission to base station """
    """ 
    filename = "../mavlink_communication_c_uart/YAW_r_des.txt"
    readdata = []
    with open(filename) as f:
        for line in f:
            readdata.append([float(n) for n in line.strip().split(',')])
            f.close
            pix_data = readdata[0]

    #print(pix_data)
    yaw_rad = pix_data[0]
    wp_des = [pix_data[1], pix_data[2]]
    
    
    #runtime =0
    #yaw_rad = 1
    #wp_des = [1, 2]
    data_list = [runtime, k, EKF.get_x_est(), EKF.get_z_meas(), EKF.get_y_est(), yaw_rad, wp_des]
    
    soc_client.soc_process_server_request(data_list)

    #print(k)
    looptime = time.time() - t1
    print(looptime)
    """
