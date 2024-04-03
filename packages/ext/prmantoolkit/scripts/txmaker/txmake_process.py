#!/usr/bin/env python
from __future__ import print_function
#
#
#	Dexter CG-Supervisor
#
#		Sanghun Kim, rman.td@gmail.com
#
#	rman.td 2019.12.12 $5
#

from PySide2 import QtWidgets, QtGui, QtCore
import os, sys
from subprocess import *

class txmake_thread(QtCore.QThread):
    increment = QtCore.Signal(str)

    def __init__(self, parent, files, opts):
        super(txmake_thread, self).__init__(parent)
        self.cancel = False
        self.files  = files # list
        self.opts   = opts  # Class TxOptions

    def run(self):
        for i in range(len(self.files)+1):
            if self.cancel:
                break
            label = ''
            if i != 0:
                label = self.files[i-1] # filename
                if self.opts.m_maptype == 'envlatl' or label.split('.')[-1] == 'hdr':
                    self.envtxmake_process(label)
                else:
                    self.txmake_process(label)
            self.increment.emit(label)

    def txmake_process(self, filename):
        outfilename = self.opts.getOutFilename(filename)
        commands  = self.opts.getCommand()
        commands += [filename, outfilename]
        cmd  = ' '.join(commands)
        pipe = Popen(cmd, shell=True)
        pipe.wait()

    def envtxmake_process(self, filename):
        commands  = self.opts.getEnvCommand()
        commands += [filename]
        cmd  = ' '.join(commands)
        pipe = Popen(cmd, shell=True)
        pipe.wait()

    @QtCore.Slot(str)
    def cancel(self):
        self.cancel = True



