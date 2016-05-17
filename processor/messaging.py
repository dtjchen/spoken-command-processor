import os
import socket


# Host for all processes
HOST = 'localhost'

# Port of this process
PORT = os.environ['SPEECH_PORT']

def send_message(message_str, destination_port):
    """Open a connection between PORT and destination_port, and send the
    message.
    """
    s = socket.socket()
    s.connect((HOST, PORT))

    # Send the message
    print('sent message to %d: %s' % (destination_port, message_str))

def listen():
    """Listen on PORT for new connections on a continuous basis. Accept them
    and print their messages.
    """
    pass


if __name__ == '__main__':
    listen()
