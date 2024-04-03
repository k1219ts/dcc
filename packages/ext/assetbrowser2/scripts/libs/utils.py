#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import webbrowser

from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

def create_blank_image(width=320, height=240, status=""):
    """가로와 세로의 크기를 지정하면 검정 이미지의 QPixmap을 반환합니다.

    Args:
        width (int): 가로 이미지의 크기
        height (int): 세로 이미지의 크기

    Returns:
        QtGui.QPixmap()

    """
    color = QtGui.QColor(0, 0, 0) # black
    pixmap = QtGui.QPixmap(QtCore.QSize(width, height))
    pixmap.fill(color)
    if status:
        painter = QtGui.QPainter(pixmap)

        if status == "Publish":
            color_status = QtGui.QColor("#1F91D0")
        elif status == "Delete":
            color_status = QtGui.QColor("#f88070")
        else:
            color_status = QtGui.QColor(color)

        brush = QtGui.QBrush(color_status)
        painter.fillRect(QtCore.QRect(0, 0, int(width*0.02), height), brush)

        painter.end()


    return pixmap

def error_message(text, title="Error", button=QtWidgets.QMessageBox.Ok):
    msg_box = QtWidgets.QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setStandardButtons(button)
    return msg_box.exec_()

def get_posix_file(file_id):
    # osascript -e 'get posix path of posix file "file:///.file/id=6571367.12937245248" -- kthxbai'
    command = """osascript -e 'get posix path of posix file "{}" -- kthxbai'""".format(file_id)
    return os.popen(command).read()

def get_grid_size(num, width=320, height=360):
    width = width-5
    total_area = width * height
    ratio = 3.0/4.0

    this_width = width
    process_area = 0
    mode = True
    while mode:

        break_mode = False
        for i in range(num):
            this_area = this_width * (this_width * ratio)
            process_area += this_area
            if total_area < process_area:
                break_mode = True
                break

        if break_mode:
            mode = True
            process_area = 0
            this_width = this_width / 2
        else:
            mode = False

    this_height = this_width*ratio

    return QtCore.QSize(this_width, this_height)

def open_file(path):
    if os.path.isfile(path):
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif sys.platform == 'win32':
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        else:
            subprocess.Popen(['xdg-open', path])
    else:
        raise OSError("The file {} does not exist.".format(path))

def open_directory(path):
    if os.path.isdir(path):
        if sys.platform == 'darwin':
            subprocess.Popen(['open', '-a', 'Finder', path])
        elif sys.platform == 'win32':
            webbrowser.open(path)
        else:
            subprocess.Popen(['xdg-open', path])
    else:
        raise OSError("The directory {} does not exist.".format(path))

def resize_pixmap(path, status, width=180, height=135, transformMode=QtCore.Qt.FastTransformation):

    new_path = replace_path(path)

    color = QtCore.Qt.black

    if not new_path:
        return create_blank_image(width, height, status)

    pixmap = QtGui.QPixmap(new_path)
    if pixmap.isNull():
        print("[DEBUG] Pixmap is NULL. {}".format(new_path))
        return create_blank_image(width, height, status)

    new_pixmap = QtGui.QPixmap(width, height)
    new_pixmap.fill(color)
    painter = QtGui.QPainter(new_pixmap)

    # scaled
    pixmap = pixmap.scaled(
        width, height, QtCore.Qt.KeepAspectRatio, transformMode) # or QtCore.Qt.SmoothTransformation

    x = 0
    y = 0
    if pixmap.width() < width:
        x = int((width - pixmap.width()) / 2.0)
    if pixmap.height() < height:
        y = int((height - pixmap.height()) / 2.0)

    painter.drawPixmap(x, y, pixmap.width(), pixmap.height(), pixmap)

    if status:
        if status == "Publish":
            color_status = QtGui.QColor("#1F91D0")
        elif status == "Delete":
            color_status = QtGui.QColor("#f88070")
        else:
            color_status = QtGui.QColor(color)

        brush = QtGui.QBrush(color_status)
        painter.fillRect(QtCore.QRect(0, 0, int(width*0.02), height), brush)

    painter.end()

    return new_pixmap

def is_unicode(unicode_or_bytestring):
    if sys.version_info < (3, 0, 0):
        return isinstance(unicode_or_bytestring, basestring)
    else:
        return isinstance(unicode_or_bytestring, str)


def replace_path(path):
    if path != '' and not os.path.exists(path):

        if sys.platform == "darwin":
            return "/opt/{}".format(path)
        elif sys.platform == "win32":
            pass
        else:
            pass

    return path


if __name__ == "__main__":
    # get_posix_file("file:///.file/id=6571367.12937245248")
    get_grid_size(2)
