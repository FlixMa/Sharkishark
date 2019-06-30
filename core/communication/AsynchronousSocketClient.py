import socket
import threading
from ..parsing import Parser


def receive(sock, debug=False):
    data = bytearray()
    while 1:
        try:
            data += sock.recv(1)

        except socket.error:
            message = bytes(data).decode('utf-8')
            if Parser.check_done_receiving(message):
                return message

        except Exception as ex:
            print('exception in receive: ', repr(ex))
            print('data on exception:', repr(bytes(data).decode('utf-8')))
            raise ex


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

    def wait_until_stopped(self):
        while self.isAlive() and not self._stop_event.wait(0.5):
            pass


class AsynchronousSocketClient(StoppableThread):

    def __init__(self, host, port):
        super().__init__()

        self.sock = socket.socket()
        self.sock.connect((host, port))

    def run(self):
        self.sock.setblocking(0)
        with self.sock:
            while not self.is_stopped():
                try:
                    message = receive(self.sock)
                    self.onMessage(message)
                except socket.error as ex:
                    self.stop()
                    print("remote peer closed the socket:", ex)
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
