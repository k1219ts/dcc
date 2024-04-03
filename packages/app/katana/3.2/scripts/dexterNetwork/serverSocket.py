from Katana import QtCore

# SERVER
import socket
import time

# serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serverSocket.bind(("localhost", 7031))
# serverSocket.listen(1)
# conn, addr = serverSocket.accept()
#
# data = conn.recv(1024)
# print data
# eval(data)


class CommandThread(QtCore.QThread):
    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)

        self.isSocketRunning = False

    def run(self):
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bindSuccess = False
        try:
            serverSocket.bind(("localhost", 7031))
            bindSuccess = True
        except:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("localhost", 7031))
            sock.sendall("killSocket")
            sock.close()
        finally:
            if not bindSuccess:
                time.sleep(1)
                serverSocket.bind(("localhost", 7031))
        print "# Bind okay"
        serverSocket.listen(1)
        self.isSocketRunning = True

        print "# Listen..."
        while self.isSocketRunning:
            conn, addr = serverSocket.accept()
            while self.isSocketRunning:
                data = conn.recv(1024)
                if not data: break

                if "killSocket" in data:
                    print "socket close"
                    serverSocket.close()
                    serverSocket = None
                    return

                try:
                    exec(data)
                except Exception as e:
                    print e.message
        serverSocket.close()
        serverSocket = None
