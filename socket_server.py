import socket


class SocServer(object):
    def __init__(self, ip_server, port_server):
        self.__server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__ip_server = ip_server
        self.__port_server = port_server

        self.__server.bind((self.__ip_server, self.__port_server))
        self.__server.listen(1)
        print('Server: Started listening on ip: ' + self.__ip_server + ' on port: ' + str(self.__port_server))

        print('...\nWaiting for client to connect...')
        self.__incoming_client, self.__incoming_address = self.__server.accept()
        print('Server: Got connection from ip: ' + self.__incoming_address[0] + ' on port: ' + self.__incoming_address[1])

    def soc_get_data_from_client(self):

        new_data = self.__incoming_client.recv(1024)
        if new_data == 'disconnect':
            print('Received command to disconnect!')
            self.__server.close()
        else:
            print('Server: Received data: ' + new_data)

        return new_data


class SocClient(object):
    def __init__(self, ip_server, port_server):
        self.__client = socket.socket()
        self.__ip_server = ip_server
        self.__port_server = port_server

        self.__client.connect((self.__ip_server, self.__port_server))

    def soc_send_data_so_server(self, data):

        self.__client.send(data)
