import dexterNetwork.serverSocket as serverSocket
from Katana import QtWidgets, Configuration

global thread
if Configuration.get('KATANA_UI_MODE'):
    thread = serverSocket.CommandThread(QtWidgets.QApplication.activeWindow())
    thread.start()
