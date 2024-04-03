#encoding=utf-8
#!/usr/bin/env python

#-------------------------------------------------------------------------------
#
#   Dexter CG Supervisor
#
#		sanghun.kim		rman.td@gmail.com
#
#	2017.03.25	$2
#-------------------------------------------------------------------------------

from Qt import QtCore, QtGui, QtWidgets, load_ui


def setup_ui( uifile, base_instance=None ):
    ui = load_ui( uifile )
    if not base_instance:
        return ui
    else:
        for member in dir(ui):
            if not member.startswith('__') and member is not 'staticMetaObject':
                setattr( base_instance, member, getattr(ui, member) )
        return ui


#	MessageBox
def messageBox( Title = 'Warning',
                Message = 'warning message',
                Icon = 'warning',	# warning, question, information, critical
                Button = ['OK', 'Cancel'],
                bgColor = [.5,.5,.5] ):
    msg = cmds.confirmDialog( title=Title,
                              message='%s    ' % Message,
                              messageAlign='center',
                              icon=Icon,
                              button=Button,
                              backgroundColor = bgColor )
    return msg

