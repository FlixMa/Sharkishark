import socket
import threading

def receive(sock):
    data = bytearray()
    try:
        while 1:
            b = sock.recv(1)
            if not b:
                if len(data) == 0:
                    raise socket.error('no data received')
                else:
                    print('last data:', repr(bytes(data).decode('utf-8')))
                    break
            data.append(b[0])
    except socket.timeout:
        pass
    except Exception as ex:
        print('exception in receive: ', repr(ex))
        print('data on exception:', repr(bytes(data).decode('utf-8')))
        raise ex
    return bytes(data).decode('utf-8')


class StoppableThread(threading.Thread):
    '''
        Thread class with a stop() method. The thread itself has to check regularly for the stopped() condition.
    '''

    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()



class AsynchronousSocketClient(StoppableThread):

    def __init__(self, host, port):
        super().__init__()

        self.sock = socket.socket()
        self.sock.connect((host, port))

    def run(self):
        self.sock.settimeout(0.5)
        with self.sock:
            while not self.is_stopped():
                try:
                    message = receive(self.sock)
                    if message != '':
                        self.onMessage(message)
                except socket.error:
                    self.stop()
                    print("remote peer closed the socket!")
                except Exception as ex:
                    print(ex)
                    raise ex

    def onMessage(self, message):
        print('-> onMessage(%s)' % message)

    def send(self, message):
        if not self.is_stopped():
            self.sock.send(message.encode('utf-8'))

    def stop(self):
        super().stop()

        while not self.is_stopped():
            pass

        self.sock.close()
