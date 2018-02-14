import socket
import pickle
import time


"""

Socket Server Class

"""


class SocServer(object):
    def __init__(self, ip_server, port_server):
        self.__server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__ip_server = ip_server
        self.__port_server = port_server
        self.__numofsentmsg = 0  # number of sent messages

        self.__server.bind((self.__ip_server, self.__port_server))
        self.__server.listen(5)
        print('Server: Started listening on ip: ' + str(self.__ip_server) + ' on port: ' + str(self.__port_server))
        self.soc_wait_and_connect_to_client()

    def soc_wait_and_connect_to_client(self):
        print('...\nWaiting for client to connect...')
        self.__conn2client, self.__incoming_address = self.__server.accept()
        print('Server: Got connection from ip: ' + str(self.__incoming_address[0]) + ' on port: ' + str(self.__incoming_address[1]))

    def soc_close_restart_connection(self):
        self.__conn2client.close()
        print('Connection lost...')
        print('Restart...')
        self.soc_wait_and_connect_to_client()
        return True

    def soc_send_data_request(self):
        self.__conn2client.send('req_data')
        # print('req_data_sent')
        while True:
            received_data = self.soc_get_data_from_client()
            if received_data is not '':
                break
        return received_data

    def soc_send_data_to_client(self, data_list):
        self.__numofsentmsg = self.__numofsentmsg + 1
        pickled_list = pickle.dumps(data_list)
        msg = pickled_list
        # msg = 'c' + str(self.__numofsentmsg) + ':' + str(data)
        self.__conn2client.send(msg)

    def soc_get_data_from_client(self):
        while True:
            try:
                msg_buf = self.__conn2client.recv(4096)
                # print('new msg form server: ' + new_msg)
            except socket.error, e:
                print "Error receiving data: %s" % e
                return False
            else:
                new_data_list = pickle.loads(msg_buf)
                return new_data_list


"""

Socket Client Class

"""


class SocClient(object):
    def __init__(self, ip_server, port_server, timeout_request=10):
        self.__ip_server = ip_server
        self.__port_server = port_server
        self.__numofsentmsg = 0  # number of sent messages
        self.__timeout = timeout_request
        self.__start_time = time.time()
        self.__timout_timer = 0
        self.soc_connect_to_server()

    def soc_connect_to_server(self):
        self.__client = socket.socket()
        try:
            self.__client.connect((self.__ip_server, self.__port_server))
        except socket.error, e:
            print "Error receiving data: %s" % e
            print('Client: connection to server failed!')
            return False
        return True

    def soc_check_timout_reached(self):
        if self.__timout_timer > self.__timeout:
            print('Client: timeout reached: ' + str(self.__timout_timer) + '> max timeout: ' + str(self.__timeout))
            return True
        else:
            return False

    def soc_close_and_reconnect(self):
        print('Client: close connection...')
        self.__client.close()
        print('Client: reconnect...')
        if self.soc_connect_to_server():
            print('Client: reconnected to server at ' + self.__ip_server + ':' + str(self.__port_server))
        return True

    def soc_send_data_to_server(self, data_list):
        self.__numofsentmsg = self.__numofsentmsg + 1
        # print(data_list)
        pickled_list = pickle.dumps(data_list)
        msg = pickled_list
        self.__client.send(msg)
        # print(msg)

        return True

    def soc_wait_for_data_request(self):
        new_msg = self.__client.recv(4096)
        if new_msg == 'req_data':
            return True
        else:
            return False

    def soc_process_server_request(self, newdata_list):

        if self.soc_wait_for_data_request():  # received_msg == 'req_data':
            self.soc_send_data_to_server(newdata_list)
            self.__start_time = time.time()
            #print('Client: data sent')
            return True
        else:
            print('Client: no data request')
            self.__timout_timer = time.time() - self.__start_time
            if self.soc_check_timout_reached():
                self.soc_close_and_reconnect()

            return False




