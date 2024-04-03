from functools import partial

from PySide2 import QtWidgets, QtCore
from maya import cmds

from dwpicker.optionvar import (
    save_optionvar, LAST_COMMAND_LANGUAGE, SEARCH_FIELD_INDEX,
    SHAPES_FILTER_INDEX)


SEARCH_AND_REPLACE_FIELDS = 'Targets', 'Label', 'Command', 'Image path'
SHAPES_FILTERS = 'All shapes', 'Selected shapes'


def warning(title, message, parent=None):
    return QtWidgets.QMessageBox.warning(
        parent,
        title,
        message,
        QtWidgets.QMessageBox.Ok,
        QtWidgets.QMessageBox.Ok)


def question(title, message, parent=None):
    result = QtWidgets.QMessageBox.question(
        parent, title, message,
        QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
        QtWidgets.QMessageBox.Ok)
    return result == QtWidgets.QMessageBox.Ok


class NamespaceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(NamespaceDialog, self).__init__(parent=parent)
        self.setWindowTitle('Select namespace ...')
        self.namespace_combo = QtWidgets.QComboBox()
        self.namespace_combo.setEditable(True)
        namespaces = cmds.namespaceInfo(listOnlyNamespaces=True, recurse=True)
        self.namespace_combo.addItems(namespaces)

        self.ok = QtWidgets.QPushButton('Ok')
        self.ok.released.connect(self.accept)
        self.cancel = QtWidgets.QPushButton('Cancel')
        self.cancel.released.connect(self.reject)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.ok)
        self.button_layout.addWidget(self.cancel)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.namespace_combo)
        self.layout.addLayout(self.button_layout)

    @property
    def namespace(self):
        return self.namespace_combo.currentText()


class CommandButtonDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(CommandButtonDialog, self).__init__(parent=parent)
        self.setWindowTitle('Create command button')
        self.label = QtWidgets.QLineEdit()

        self.python = QtWidgets.QRadioButton('Python')
        self.mel = QtWidgets.QRadioButton('Mel')
        self.language = QtWidgets.QWidget()
        self.language_layout = QtWidgets.QVBoxLayout(self.language)
        self.language_layout.setContentsMargins(0, 0, 0, 0)
        self.language_layout.addWidget(self.python)
        self.language_layout.addWidget(self.mel)

        self.language_buttons = QtWidgets.QButtonGroup()
        self.language_buttons.buttonReleased.connect(self.change_state)
        self.language_buttons.addButton(self.python, 0)
        self.language_buttons.addButton(self.mel, 1)

        self.command = QtWidgets.QPlainTextEdit()

        self.options_layout = QtWidgets.QFormLayout()
        self.options_layout.addRow('Label: ', self.label)
        self.options_layout.addRow('Language: ', self.language)
        self.options_layout.addRow('Command: ', self.command)

        self.ok = QtWidgets.QPushButton('Ok')
        self.ok.released.connect(self.accept)
        self.cancel = QtWidgets.QPushButton('Cancel')
        self.cancel.released.connect(self.reject)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.ok)
        self.button_layout.addWidget(self.cancel)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.options_layout)
        self.layout.addLayout(self.button_layout)

        self.set_ui_states()

    def set_ui_states(self):
        index = cmds.optionVar(query=LAST_COMMAND_LANGUAGE)
        button = self.language_buttons.button(index)
        button.setChecked(True)

    @property
    def values(self):
        language = 'python' if self.python.isChecked() else 'mel'
        return {
            'action.left.language': language,
            'text.content': self.label.text(),
            'action.left.command': self.command.toPlainText(),}

    def change_state(self, *_):
        save_optionvar(
            LAST_COMMAND_LANGUAGE,
            self.language_buttons.checkedId())


class SearchAndReplaceDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SearchAndReplaceDialog, self).__init__(parent=parent)
        self.setWindowTitle('Search and replace in shapes')
        self.sizeHint = lambda: QtCore.QSize(320, 80)

        self.filters = QtWidgets.QComboBox()
        self.filters.addItems(SHAPES_FILTERS)
        self.filters.setCurrentIndex(cmds.optionVar(query=SHAPES_FILTER_INDEX))
        function = partial(save_optionvar, SHAPES_FILTER_INDEX)
        self.filters.currentIndexChanged.connect(function)
        self.fields = QtWidgets.QComboBox()
        self.fields.addItems(SEARCH_AND_REPLACE_FIELDS)
        self.fields.setCurrentIndex(cmds.optionVar(query=SEARCH_FIELD_INDEX))
        function = partial(save_optionvar, SEARCH_FIELD_INDEX)
        self.fields.currentIndexChanged.connect(function)
        self.search = QtWidgets.QLineEdit()
        self.replace = QtWidgets.QLineEdit()

        self.ok = QtWidgets.QPushButton('Replace')
        self.ok.released.connect(self.accept)
        self.cancel = QtWidgets.QPushButton('Cancel')
        self.cancel.released.connect(self.reject)

        self.options = QtWidgets.QFormLayout()
        self.options.setContentsMargins(0, 0, 0 , 0)
        self.options.addRow('Apply on: ', self.filters)
        self.options.addRow('Field to search: ', self.fields)
        self.options.addRow('Search: ', self.search)
        self.options.addRow('Replace by: ', self.replace)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(0, 0, 0 , 0)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.ok)
        self.button_layout.addWidget(self.cancel)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addLayout(self.options)
        self.layout.addLayout(self.button_layout)

    @property
    def field(self):
        '''
        0 = Targets
        1 = Label
        2 = Command
        3 = Image path
        '''
        return self.fields.currentIndex()

    @property
    def filter(self):
        '''
        0 = Apply on all shapes
        1 = Apply on selected shapes
        '''
        return self.filters.currentIndex()