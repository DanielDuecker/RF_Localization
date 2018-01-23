import socket
import pickle


class SocServer(object):
    def __init__(self, ip_server, port_server):
        self.__server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__ip_server = ip_server
        self.__port_server = port_server
        self.__numofsentmsg = 0  # number of sent messages

        self.__server.bind((self.__ip_server, self.__port_server))
        self.__server.listen(1)
        print('Server: Started listening on ip: ' + str(self.__ip_server) + ' on port: ' + str(self.__port_server))

        print('...\nWaiting for client to connect...')
        self.__incoming_client, self.__incoming_address = self.__server.accept()
        print('Server: Got connection from ip: ' + str(self.__incoming_address[0]) + ' on port: ' + str(self.__incoming_address[1]))

    def soc_send_data_to_client(self, data_list):
        self.__numofsentmsg = self.__numofsentmsg + 1
        # print(data_list)
        pickled_list = pickle.dumps(data_list)
        msg = pickled_list
        self.__incoming_client.send(msg)
        # print(msg)

        return True
    """
    def soc_get_data_from_client(self):
        new_msg = self.__incoming_client.recv(8096)
        print('server: rawdata ' + str(new_msg))
        new_data_list = pickle.loads(new_msg)
        print('Server: Received data: ' + str(new_data_list))
        return new_data_list
    """
    def soc_wait_for_data_request(self):
        new_msg = self.__incoming_client.recv(8096)
        print('wait for request')
        if new_msg == 'req_data':
            return True
        else:
            return False

    def soc_process_request(self, newdata_list):

        if self.soc_wait_for_data_request():  # received_msg == 'req_data':
            self.soc_send_data_to_client(newdata_list)
            print('data sent')
            return True
        else:
            return False


class SocClient(object):
    def __init__(self, ip_server, port_server):
        self.__client = socket.socket()
        self.__ip_server = ip_server
        self.__port_server = port_server
        self.__numofsentmsg = 0  # number of sent messages

        self.__client.connect((self.__ip_server, self.__port_server))

    def soc_send_data_request(self):
        self.__client.send('req_data')
        #print('req_data_sent')
        while True:
            received_data = self.soc_get_data_from_server()
            if received_data is not '':
                break
        return received_data

    def soc_send_data_to_server(self, data_list):
        self.__numofsentmsg = self.__numofsentmsg + 1
        pickled_list = pickle.dumps(data_list)
        msg = pickled_list
        #msg = 'c' + str(self.__numofsentmsg) + ':' + str(data)
        self.__client.send(msg)

    def soc_get_data_from_server(self):
        while True:
            new_msg = self.__client.recv(8096)
            #print('new msg form server: ' + new_msg)
            if new_msg is not '':
                new_data_list = pickle.loads(new_msg)
                break
        #print('Client: Received data: ' + str(new_data_list))
        return new_data_list
