import os
import socket
import sys

# Host for all processes
HOST = 'localhost'

# Port of this process
PORT = os.environ['SPEECH_PORT']

def send_message(message_str, destination_port):
    """Open a connection between PORT and destination_port, and send the
    message.
    """
    # Create a TCP/IP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    destination_addr = (HOST, destination_port)
    try: # Connect and send message
        s.connect(destination_addr)
        s.sendall(message_str)
    except socket.error as err:
        print('Error Code : %s, Message %s' % (str(err[0]), err[1]))
    finally:
        s.close()

    # Send the message
    print('sent message to %d: %s' % (destination_port, message_str))

    return True

def listen():
    """Listen on PORT for new connections on a continuous basis. Accept them
    and print their messages.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        # Bind socket to address
        s.bind((HOST, PORT))

        # Start listening
        backlog = 10
        s.listen(backlog)

        # Block to accept connection
        recv_bytes = 1024
        while True:
            conn, addr = s.accept()
            message = conn.recv(recv_bytes)

            # Print message
            print('Message (%d:%d): %s' % (addr[0], addr[1], message))

            conn.close()

    except socket.error as err:
        print('Error Code : %d, Message %s' % (err[0], err[1]))
        sys.exit()
    finally:
        s.close()

if __name__ == '__main__':
    listen()
