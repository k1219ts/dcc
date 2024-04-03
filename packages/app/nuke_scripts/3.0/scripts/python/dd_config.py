# -*- coding: utf-8 -*-
from PySide2 import QtWidgets, QtCore, QtGui

import subprocess
import threading
import json
from tactic_client_lib import TacticServerStub

class EmailThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.info = None

    def run(self):
        server = TacticServerStub( login='taehyung.lee', password='dlxogud', server='10.0.0.51', project='show1' )
        shot_exp = "@SOBJECT(sthpw/login['department_short','NEQ','None|TES|VEN|CLT|ZZZ|DMM'])"
        self.info = server.eval( shot_exp )

class GlobalConfigImpl(QtCore.QObject):
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.mailInfo = None

        self.engDic = {}
        self.korDic = {}
        self.teamDic = {}

        threadObject = EmailThread()
        threadObject.start()
        threadObject.join()

        for i in threadObject.info:
            self.engDic[i['login']] = i
            self.korDic[i['name_kr']] = i
            if self.teamDic.get(i['department_short']):
                self.teamDic[i['department_short']].append(i['login'])
            else:
                self.teamDic[i['department_short']]= [i['login']]

        self.prjConfigPath = '/dexter/Cache_DATA/comp/TD_hslth/Global_File/prj_seq.json'
        self.prjConfigData = json.loads(open(self.prjConfigPath, 'r').read())
        #self.prjName_inv = {v:k for k,v in self.prjConfigData['prj_name'].items()}


        self.colorScheme = {'Waiting':QtGui.QColor('#D7D7D7'), 'Omit':QtGui.QColor('#707070'),
                            'Hold':QtGui.QColor('#9D9D9D'), 'Ready':QtGui.QColor('#F7F6CE'),
                            'In-Progress':QtGui.QColor('#CAE1CA'), 'Retake': QtGui.QColor('#E17F81'),
                            'Review':QtGui.QColor('#A0E6B0'), 'OK':QtGui.QColor('#83D8DE'),
                            'Approved':QtGui.QColor('#6F8BCA'), '':QtGui.QColor()}

        self.statusOrder = {'Approved':0, 'OK': 1, 'Review': 2, 'In-Progress':3,
                            'Ready':4, 'Retake': 5, 'Waiting': 6, 'Hold':7,
                            'Omit':8}

        self.teamColor = {'matchmove':QtGui.QColor('#DA8CDD'), 'model':QtGui.QColor('#978CDD'),
                          'creature':QtGui.QColor('#8CA9DD'), 'animation':QtGui.QColor('#8CCDDD'),
                          'texture':QtGui.QColor('#8CDDC5'), 'lighting':QtGui.QColor('#8CDD9B'),
                          'mattepaint':QtGui.QColor('#BFDD8C'), 'fx':QtGui.QColor('#DDC38C'),
                          'comp':QtGui.QColor('#DD8C8C'), 'rnd':QtGui.QColor('#8B9AA5'),
                          'previz':QtGui.QColor('#B0BBDA'), 'edit':QtGui.QColor('#DCD2C3'),
                          'di':QtGui.QColor('#DCC3C3')}



class GlobalConfig():
    __instance = None
    @staticmethod
    def instance():
        if not GlobalConfig.__instance:
            GlobalConfig.__instance = GlobalConfigImpl()
        return GlobalConfig.__instance
