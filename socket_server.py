import socket
import sys
from threading import *
"""
host = ''
port = 50009
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((host, port))
except socket.error as e:
    print(str(e))

s.listen(5)

def threaded_client(conn):
    conn.send('Type your info!\n')

    while True:
        data = conn.recv(2048)
        reply = 'Server answer: ' + data
        if not data:
            break
        conn.sendall(reply)
    conn.close()

while True:
    conn, addr = s.accept()
    print('connected to: ' + str(addr[0]) + ' on port ' + str(addr[1]))

    start_new_thread(threaded_client,(conn,))

    print('huhu')
"""


class SocServer(object):
    def __init__(self, ip_server, port_server):
        self.__server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__ip_server = ip_server
        self.__port_server = port_server

        self.__server.bind((self.__ip_server, self.__port_server))
        self.__server.listen(1)
        print('Server: Started listening on ip: ' + str(self.__ip_server) + ' on port: ' + str(self.__port_server))

        print('...\nWaiting for client to connect...')
        self.__incoming_client, self.__incoming_address = self.__server.accept()
        print('Server: Got connection from ip: ' + str(self.__incoming_address[0]) + ' on port: ' + str(self.__incoming_address[1]))

    def soc_send_data_to_client(self, data):

        self.__incoming_client.send(data)

        return True

    def soc_get_data_from_client(self):

        new_data = self.__incoming_client.recv(1024)
        if new_data == 'disconnect':
            print('Received command to disconnect!')
            self.__server.close()
        else:
            print('Server: Received data: ' + str(new_data))

        return new_data


class SocClient(object):
    def __init__(self, ip_server, port_server):
        self.__client = socket.socket()
        self.__ip_server = ip_server
        self.__port_server = port_server

        self.__client.connect((self.__ip_server, self.__port_server))

    def soc_send_data_to_server(self, data):

        self.__client.send(data)

    def soc_get_data_from_server(self):

        new_data = self.__client.recv(1024)
        print('Client: Received data: ' + str(new_data))

        return new_data
