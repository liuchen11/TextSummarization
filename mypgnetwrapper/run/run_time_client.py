import sys
import socket
import select

host='127.0.0.1'
port=5001

socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
socket.settimeout(2)

try:
    socket.connect((host,port))
except:
    print('Unable to connect')
    exit(1)

print('Connected to the remote host, you can now type in a URL')
print('>>>')

while True:
    socket_list=[socket,sys.stdin]

    ready2read,ready2write,in_error=select.select(socket_list,[],[])

    for sock in ready2read:
        if sock==socket:
            data=sock.recv(65536)
            if not data:
                print('Disconnect from the remost host')
                exit(0)
            else:
                sys.stdout.write(data)
        else:
            message=sys.stdin.readline()
            socket.send(message)
        print('>>>')
        sys.stdout.flush()
