# -*- coding: utf-8 -*-
import db_query, getpass
#from PyQt4 import QtCore, QtGui

from pymodule import Qt
from pymodule.Qt import QtWidgets
from pymodule.Qt import QtGui
from pymodule.Qt import QtCore

class ConfigThread(QtCore.QThread):
    emitMessage = QtCore.Signal(object)

    def __init__(self):
        QtCore.QThread.__init__(self)
        self.finished.connect(self.queryDone)

    def run(self):
        # CORE ELEMENT
        # ------------------------------------------------------------------------#
        print "START CORE ELEMENT"
        self.structure = {'CORE ELEMENT':{}}

        for coreType in db_query.getCoreElement():
            self.structure['CORE ELEMENT'][coreType['name_kr']] = {}
            self.structure['CORE ELEMENT'][coreType['name_kr']]['searchTerm'] = {"$text": {"$search": coreType['searchTerm'],
                                                                                "$caseSensitive": False},
                                                                      "enabled": True,
                                                                      'type': 'COMP_SRC'
                                                                      }

        print "END CORE ELEMENT"
        # ------------------------------------------------------------------------#

        # DEFAULT TYPE MENU
        for assetType in db_query.getDistinct("type"):
            self.structure[assetType] = {}
            self.structure[assetType]['searchTerm'] = {'type':assetType,
                                  'enabled': True}

            for prj in db_query.getFindDistinct({'type':assetType},
                                                'project'):
                self.structure[assetType][prj] = {}
                subCategory = db_query.getFindDistinct({'type': assetType,
                                                        'project': prj,
                                                        'enabled': True},
                                                       'category')


                if subCategory:
                    for cat in subCategory:
                        self.structure[assetType][prj][cat]  = {}

                        self.structure[assetType][prj][cat]['searchTerm'] = {'type':assetType,
                                                                             'project': prj,
                                                                             'category': cat,
                                                                             'enabled': True}
                else:
                    self.structure[assetType][prj]['searchTerm'] = {'type':assetType,
                                                                    'project':prj,
                                                                    'enabled':True}



        # ------------------------------------------------------------------------#

    def __del__(self):
        self.wait()

    def queryDone(self):
        self.emitMessage.emit(self.structure)


class GlobalConfigImpl(QtCore.QObject):
    def __init__(self):
        QtCore.QObject.__init__(self)
        self.config_dic = db_query.getUserConfig()
        if not(self.config_dic):
            self.config_dic = {}
            self.config_dic['play'] = 'click'
            self.config_dic['auto_repeat'] = True
            self.config_dic['item_per_click'] = 20
            self.config_dic['bookmark'] = []
            self.config_dic['play_speed'] = 3.0

        if not(self.config_dic.has_key('play')):
            self.config_dic['play'] = 'click'

        if not (self.config_dic.has_key('auto_repeat')):
            self.config_dic['auto_repeat'] = True

        if not (self.config_dic.has_key('item_per_click')):
            self.config_dic['item_per_click'] = 20

        if not (self.config_dic.has_key('bookmark')):
            self.config_dic['bookmark'] = []

        if not (self.config_dic.has_key('play_speed')):
            self.config_dic['play_speed'] = 3.0

    def update(self, dic):
        self.config_dic = dic
        db_query.updateUserConfig(getpass.getuser(), self.config_dic)

    def getConfig(self):
        return self.config_dic


class GlobalConfig():
    __instance = None

    @staticmethod
    def instance():
        if not GlobalConfig.__instance:
            GlobalConfig.__instance = GlobalConfigImpl()
        return GlobalConfig.__instance



