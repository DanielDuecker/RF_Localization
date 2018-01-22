import socket


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

    def soc_send_data_to_client(self, data):
        self.__numofsentmsg = self.__numofsentmsg + 1
        msg = 's' + str(self.__numofsentmsg) + ':' + str(data)
        self.__incoming_client.send(msg)
        # print(msg)

        return True

    def soc_get_data_from_client(self):

        new_data = self.__incoming_client.recv(8096)

        if new_data == 'disconnect':
            print('Received command to disconnect!')
            self.__server.close()
        # else:
             # print('Server: Received data: ' + str(new_data))

        return new_data

    def soc_process_request(self, newdata):
        received_msg = self.soc_get_data_from_client()
        if received_msg == 'req_data':
            self.soc_send_data_to_client(newdata)
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
        while True:
            received_data = self.soc_get_data_from_server()
            if received_data is not '':
                break
        return received_data

    def soc_send_data_to_server(self, data):
        self.__numofsentmsg = self.__numofsentmsg + 1
        msg = 'c' + str(self.__numofsentmsg) + ':' + str(data)
        self.__client.send(msg)

    def soc_get_data_from_server(self):

        new_data = self.__client.recv(8096)
        print('Client: Received data: ' + str(new_data))
        return new_data
