#!/usr/bin/env python
import pygtk

pygtk.require('2.0')

import gtk
import os
import Database


def alert(msg):
    '''Show a dialog with a simple message.'''
    dialog = gtk.MessageDialog()
    dialog.set_markup(msg)
    dialog.run()


def main():
    test = os.environ.get('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS', '')

    files = []
    for file in test.split('\n'):
        if not file == "":
            files.append(file)

    for filePath in files:
        # alert("filePath : %s" % filePath)
        try:
            if '/assetlib/Texture/' in filePath:
                retMsg = Database.AddSCItem(filePath)

            elif '/assetlib/3D' in filePath:
                retMsg = Database.AddItem(filePath)
            else:
                pass

            if retMsg:
                alert(retMsg)

        except Exception as e:
            alert(e.message)


if __name__ == '__main__':
    main()

# #!/bin/python
# # nautilus script
#
# import os
# import sys
# from . import Database
#
# def _Msg(msg):
#     sys.stdout.write(msg + '\n')
#
# def _Err(msg):
#     sys.stderr.write(msg + '\n')
#
# def debugPrint(status, msg):
#     if status == 'error':
#         color = '[1;31m'
#     else:
#         color = '[1;34m'
#     sys.stdout.write('\033%s' % color)
#     sys.stdout.write(msg)
#     sys.stdout.write('\n')
#     sys.stdout.write('\033[0;0m')
#
# if __name__ == '__main__':
#     debugPrint("info", "Hello World")
#
#     filepath = os.getenv('NAUTILUS_SCRIPT_SELECTED_FILE_PATHS')
#     debugPrint("info", filepath)
#     # Database.AddItem(filepath)