#coding:utf-8
from __future__ import print_function

from PySide2 import QtWidgets, QtCore, QtGui



def ActiveUI(win):
    enable = win.ui.stu_clip_checkBox.checkState() == QtCore.Qt.Checked

    win.SetEnabled(win.ui.stu_clip_checkBox, enable, False)

    win.SetEnabled(win.ui.stu_frame_label, enable)
    win.SetEnabled(win.ui.stu_frame_start_label, enable)
    win.SetEnabled(win.ui.stu_frame_end_label, enable)

    win.SetEnabled(win.ui.stu_export_label, enable)
    win.SetEnabled(win.ui.stu_export_start_lineEdit, enable)
    win.SetEnabled(win.ui.stu_export_end_lineEdit, enable)
    win.SetEnabled(win.ui.stu_loop_checkBox, enable)

    win.SetEnabled(win.ui.stu_fps_label, enable)
    win.SetEnabled(win.ui.stu_fps_lineEdit, enable)

    win.SetEnabled(win.ui.stu_speed_m0_8_checkBox, enable)
    win.SetEnabled(win.ui.stu_speed_m1_0_checkBox, enable)
    win.SetEnabled(win.ui.stu_speed_m1_5_checkBox, enable)

    win.SetEnabled(win.ui.stu_speed_label, enable)
    win.SetEnabled(win.ui.stu_speed_lineEdit, enable)
