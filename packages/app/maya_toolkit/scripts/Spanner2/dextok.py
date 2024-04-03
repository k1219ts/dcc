# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import site
import json

import pika
import datetime
HOST = '10.0.0.13'

# import Qt
from PySide2 import QtGui
from PySide2 import QtWidgets
from PySide2 import QtCore

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
SITE = dxConfig.getHouse()

# import pymongo
from pymongo import MongoClient
client = MongoClient(DB_IP)

def sendChatMessege_db(msg, key):
    data = {}
    data['message'] = msg['text']
    data['time'] = datetime.datetime.now().isoformat()
    data['sender'] = msg['name']
    data['key'] = key
    data['image'] = msg['image']
    data['event_type'] = 'chat'
    
    db = client['PUBLISH']
    coll = db['spanner2_talk']
    coll.insert(data)  
    
def sendChatMessege(msg, key, exchangeName ):
    routing_key = key
    cred = pika.PlainCredentials('dexter', 'dexter')
    connection = pika.BlockingConnection(pika.ConnectionParameters(HOST, 5672, 'rabbitmq', cred))
    channel = connection.channel()
    channel.exchange_declare(exchange=exchangeName,
                             exchange_type='topic')

    channel.basic_publish(exchange=exchangeName,
                          routing_key=routing_key,
                          body=json.dumps(msg),
                          properties=pika.BasicProperties(delivery_mode=2, )
                          )
    connection.close()                          
                     

# DEXTER CHAT
class ChatItem(QtWidgets.QWidget):
    def __init__(self, parent=None ):
        super(ChatItem, self).__init__(parent)
        self.textLayout = QtWidgets.QVBoxLayout()
        self.name = QtWidgets.QLabel()
        self.text = QtWidgets.QLabel()
        self.text.setTextFormat(QtCore.Qt.RichText)
        self.text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.timeLabel = QtWidgets.QLabel( datetime.datetime.now().strftime("%H:%M") )
        self.timeLabel.setAlignment(QtCore.Qt.AlignBottom)
        self.textLayout.addWidget(self.name)
        self.textLayout.addWidget(self.text)
        
        self.allLayout = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLabel()
        
        self.allLayout.addWidget(self.label, 0)
        self.allLayout.addLayout(self.textLayout, 1)
        self.allLayout.addWidget(self.timeLabel, 2)
        
        self.setLayout(self.allLayout)
        
        self.name.setStyleSheet("color: white;" )
        self.text.setStyleSheet("color: black; padding: 5 5 5 10 px; border-radius:5px; background: rgb(255,255,255,100);")
        

    def setName(self, text):
        self.name.setText(text)
        
    def setText(self, text):
        self.text.setText(unicode(text))
        
    def setColor(self, color):
        self.text.setStyleSheet("color: black; padding: 5 5 5 10 px; border-radius:5px; background: %s;" %color)
        
    def setTime(self, time):
        self.timeLabel.setText( time[5:16] )
        
    def setLabel(self, path):
        if path == "":
            return
        pixmap = QtGui.QPixmap(QtCore.QSize(40,40))
        picture = QtGui.QPixmap(path)
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(0,0,40,40,QtGui.QColor(45,45,45,255))
        circle = QtGui.QPainterPath()
        circle.addEllipse(0,0,35,35)
        painter.setClipPath(circle)
        painter.drawPixmap(-2,0,40,40,picture)

        self.label.setPixmap(pixmap)   
        painter.end()

    def setLabelText(self, text='TACTIC'):
        pixmap = QtGui.QPixmap(QtCore.QSize(40, 40))
        painter = QtGui.QPainter(pixmap)
        circle = QtGui.QPainterPath()
        circle.addEllipse(0, 0, 35, 35)
        font = QtGui.QFont('Monospace', 7)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.fillRect(0, 0, 40, 40, QtGui.QColor(45, 45, 45))
        painter.fillPath(circle, QtGui.QColor(59,118,177,200))
        painter.drawText(QtCore.QRect(-2, -2, 40, 40), QtCore.Qt.AlignCenter, text[0:5])

        self.label.setPixmap(pixmap)
        painter.end()

        
# DEXTER MESSEGE
class PikaClass(QtCore.QThread):
    messageReceived = QtCore.Signal(str)
    emitMessage = QtCore.Signal(str)
    def __init__(self, parent, binding_keys, exchangeName):
        QtCore.QThread.__init__(self, parent)
        self.binding_keys = binding_keys
        self.exchangeName = exchangeName
        # AUTH
        cred = pika.PlainCredentials('dexter', 'dexter')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(HOST,
                                                                            5672,
                                                                            'rabbitmq',
                                                                            cred))
        self.createChannel()                                                                        
        self.messageReceived.connect(self.sendMessage)
        
    def createChannel(self):         
        # TACTIC ALARM                                                                   
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.exchangeName,
                                      exchange_type='topic')
                                      
        result = self.channel.queue_declare(exclusive=True, durable=True)
        self.queue_name = result.method.queue
        for bk in self.binding_keys:
            self.channel.queue_bind(exchange=self.exchangeName,
                                    queue=self.queue_name,
                                    routing_key=bk)

    def changeKey(self, binding_keys):
        result = self.channel.queue_declare(exclusive=True, durable=True)
        self.queue_name = result.method.queue
        for bk in binding_keys:
            self.channel.queue_bind(exchange=self.exchangeName,
                                    queue=self.queue_name,
                                    routing_key=bk)                                                                                       
                                
    def run(self):
        self.channel.basic_qos(prefetch_count=1)
        print ' [*] Waiting for messages. channel: %s ' %self.exchangeName
        self.channel.basic_consume(self.callback,
                                   queue=self.queue_name)
        self.channel.start_consuming()

    def __del__(self):
        self.wait()

    def sendMessage(self, body):
        text = " [x] Received %r" % (body,)
        self.emitMessage.emit(body)

    def callback(self, ch, method, properties, body):
        ch.basic_ack(delivery_tag = method.delivery_tag)
        self.messageReceived.emit(body)
        
    def deleteChannel(self):
        self.channel.queue_delete(queue=self.queue_name)   
    
class Talk_tab(QtWidgets.QWidget):        
    def __init__(self, parent=None):
        super(Talk_tab, self).__init__(parent)
        self.setObjectName("talk_tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self)
        self.gridLayout_3.setObjectName(("gridLayout_3"))
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName(("gridLayout"))
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName(("horizontalLayout"))
        self.chat_lineEdit = QtWidgets.QLineEdit(self)
        self.chat_lineEdit.setMinimumSize(QtCore.QSize(0, 30))
        self.chat_lineEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.chat_lineEdit.setObjectName(("chat_lineEdit"))
        self.horizontalLayout.addWidget(self.chat_lineEdit)
        self.pushButton = QtWidgets.QPushButton(self)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 30))
        self.pushButton.setMaximumSize(QtCore.QSize(60, 30))
        self.pushButton.setStyleSheet(("color: white;\n"
"background: rgb(59,118,177);"))
        self.pushButton.setObjectName(("pushButton"))
        self.horizontalLayout.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.talk_listWidget = QtWidgets.QListWidget(self)
        self.talk_listWidget.setObjectName(("talk_listWidget"))
        self.gridLayout.addWidget(self.talk_listWidget, 0, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(("horizontalLayout_2"))
        self.tactic_popup_checkBox = QtWidgets.QCheckBox(self)
        self.tactic_popup_checkBox.setObjectName(("tactic_popup_checkBox"))
        self.horizontalLayout_2.addWidget(self.tactic_popup_checkBox)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.gridLayout_3.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        self.pushButton.setText("send")
        self.tactic_popup_checkBox.setText("ALERT POPUP")
              
