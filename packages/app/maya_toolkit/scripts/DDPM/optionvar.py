# -*- coding: utf-8 -*-
import getpass
import logging
import maya.cmds as cmds

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def initialize():
    logger.debug(u'Get saved ui states')
    if cmds.optionVar(ex='M3_codec') == 0:
        cmds.optionVar(sv=('M3_codec', 'H.264 LT'))
    if cmds.optionVar(ex='M3_width') == 0:
        cmds.optionVar(iv=('M3_width', 1280))
    if cmds.optionVar(ex='M3_height') == 0:
        cmds.optionVar(iv=('M3_height', 720))
    if cmds.optionVar(ex='M3_depart') == 0:
        cmds.optionVar(sv=('M3_depart', "animation"))
    if not cmds.optionVar(ex='M3_autoHUD'):
        cmds.optionVar(sv=('M3_autoHUD', 'True'))
    if not cmds.optionVar(ex='M3_removeSequencer'):
        cmds.optionVar(sv=('M3_removeSequencer', 'True'))
    if not cmds.optionVar(ex='M3_offScreenVal'):
        cmds.optionVar(sv=('M3_offScreenVal', 'False'))
    if not cmds.optionVar(ex='M3_cameraSequencer'):
        cmds.optionVar(sv=('M3_cameraSequencer', 'False'))
    if not cmds.optionVar(ex='M3_addStemp'):
        cmds.optionVar(sv=('M3_addStemp', 'False'))
    if not cmds.optionVar(ex='M3_addCompRetime'):
        cmds.optionVar(sv=('M3_addCompRetime', 'False'))
    if not cmds.optionVar(ex='M3_showImageplane'):
        cmds.optionVar(sv=('M3_showImageplane', 'True'))
    if cmds.optionVar(ex='M3_User') == 0:
        cmds.optionVar(sv=('M3_User', getpass.getuser()))
    if cmds.optionVar(ex='M3_prvPreset') == 0:
        cmds.optionVar(sv=('M3_prvPreset', 'hd720'))
    if cmds.optionVar(ex='M3_project') == 0:
        cmds.optionVar(sv=('M3_project', 'test_shot'))

def setDefault():
    cmds.optionVar(iv=('M3_width', 1280))
    cmds.optionVar(iv=('M3_height', 720))
    cmds.optionVar(iv=('M3_autoHUD', 1))
    cmds.optionVar(iv=('M3_offScreen', 0))
    cmds.optionVar(sv=('M3_codec', 'H.264 LT'))
    cmds.optionVar(sv=('M3_depart', "animation"))
    cmds.optionVar(sv=('M3_User', getpass.getuser()))
    cmds.optionVar(sv=('M3_prvPreset', 'hd720'))
    cmds.optionVar(sv=('M3_project', 'test_shot'))
