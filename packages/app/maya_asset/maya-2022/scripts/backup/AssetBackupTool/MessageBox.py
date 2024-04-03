'''
'    @author    : daeseok.chae
'    @date      : 2017.02.10
'    @brief     : Simple Use MessageBox class
'''
import pymodule.Qt.QtWidgets as QtWidgets
import pymodule.Qt.QtGui as QtGui

class MessageBox():
    def __init__(self, messageText):
        messageBox = QtWidgets.QMessageBox()
        messageBox.setFont(QtGui.QFont("Cantarell", 13))
        messageBox.setText(messageText)
        
        messageBox.exec_()