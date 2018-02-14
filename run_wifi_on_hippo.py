import socket_server
import time
import estimator
import numpy as np


host_ip = '192.168.0.100'  # thinkpad ethernet
#host_ip = '192.168.0.101'  # thinkpad wifi-intern

host_port = 50008

#soc_client = socket_server.SocClient(host_ip, host_port)

""" Data Transmission to base station """
while True:     
    filename = "../mavlink_communication_c_uart/YAW_wpdes_vehpos.txt"
    readdata = []
    with open(filename) as f:
        for line in f:
            readdata.append([float(n) for n in line.strip().split(',')])
            f.close
            pix_data = readdata[0]

    filename2 = "z_y.txt"
    readdata2 = []
    with open(filename2) as f:
        for line in f:
            readdata2.append([float(n) for n in line.strip().split(',')])
            f.close
            pix_data2 = readdata2[0]

    k = 1
    yaw_rad = pix_data[0]
    wp_des = [pix_data[1], pix_data[2]]
    veh_pos = [pix_data[3], pix_data[4]]
    k = pix_data[5]
    z_meas = [pix_data2[0],pix_data2[1],pix_data2[2],pix_data2[3],pix_data2[4]]
    y_est = [pix_data2[6],pix_data2[7],pix_data2[8],pix_data2[9],pix_data2[10],pix_data2[11]]
    # soc_client.soc_process_server_request(data_list)
    data_list = [k, z_meas, y_est, yaw_rad, wp_des,veh_pos]
    print(data_list)
   
