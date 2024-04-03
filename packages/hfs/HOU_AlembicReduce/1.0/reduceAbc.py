#!/usr/bin/env python

import pygtk
pygtk.require('2.0')

import gtk
import os
import subprocess

def alert(msg):
    '''Show a dialog with a simple message.'''

    dialog = gtk.MessageDialog()
    dialog.set_markup(msg)
    dialog.run()

def main():
    test = os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS', '')
    scriptPath = '/backstage/apps/rez/RezDCC/packages/hfs/HOU_AlembicReduce/1.0/reduceTractor.py'

    files = []
    for file in test.split('\n'):
        if not file == "":
            files.append(file)

    for filePath in files:
        cmd = '/backstage/bin/DCC rez-env pylibs -- python {0} {1}'.format(scriptPath, filePath)
        # alert("filePath : %s" % filePath)
        # alert(cmd)
        os.system(cmd)
        # p = subprocess.Popen(cmd, shell=True, env = env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
    alert("spool 35 Reduce Job %s Object" % len(files))

if __name__ == '__main__':
    main()
