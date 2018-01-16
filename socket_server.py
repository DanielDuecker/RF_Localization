import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

ip = '134.28.27.35' #socket.gethostbyname(socket.gethostname())
port = 1234

address = (ip, port)



server.bind(address)
print('Server data = ' + ip + ' port = ' + str(port))
server.listen(1)


client, addr = server.accept()

print(addr)

while True:
    data = client.recv(1024)
    print("Received: " + data + " from client!")
    if (data == "Hello Server"):
        client.send("Hello Client")
        print("Processing done \n Reply sent!")
    elif(data == "disconnect"):
        client.send("Goodbye")
        client.close()
        break
    else:
        client.send("Data was invalid!")
        print("Processing done - Invalid data \n Reply sent!")

