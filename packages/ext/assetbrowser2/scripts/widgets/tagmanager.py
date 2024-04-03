#-*- coding: utf-8 -*-
from pymodule.Qt import QtCompat
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

import resources_rc
from core import Database
from widgets.ui_tagmanger import Ui_TagManager
from libs.hashableQStandardItem import HashableQStandardItem
from libs.ustr import ustr
from libs.utils import error_message

class TagManager(QtWidgets.QWidget):

    clicked = QtCore.Signal(object)
    changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(TagManager, self).__init__(parent)
        # ui_path = os.path.dirname(os.path.abspath(__file__))
        # self.ui = QtCompat.loadUi(os.path.join(ui_path, "tagmanager.ui"), self)
        self.ui = Ui_TagManager()
        self.ui.setupUi(self)

        self.tag_collection = {} # HashableQStandardItem: {tag_document}
        self.tag_dictionary = {} # _id: name
        self.tag_model = QtGui.QStandardItemModel()
        self.tag_proxy_model = QtCore.QSortFilterProxyModel()
        self.tag_proxy_model.setSourceModel(self.tag_model)

        self.ui.list_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ui.list_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.list_view.customContextMenuRequested.connect(self.pop_list_view_menu)

        self.ui.icon_mode_label.clicked.connect(self.icon_mode_clicked)
        self.ui.list_mode_label.clicked.connect(self.list_mode_clicked)
        self.ui.add_label.clicked.connect(self.add_btn_clicked)
        self.ui.update_label.clicked.connect(self.update_label_clicked)
        self.ui.list_view.clicked.connect(self.list_view_clicked)
        self.ui.filter_line_edit.textChanged.connect(self.filter_line_edit_changed)

        self.action_edit = QtWidgets.QAction("Edit", self)
        self.action_edit.triggered.connect(self.edit_btn_clicked)

        self.action_delete = QtWidgets.QAction("Delete", self)
        self.action_delete.triggered.connect(self.delete_btn_clicked)

        self.update_list_view()
        self.set_view_mode("icon")
        self.ui.update_label.hide()

    def icon_mode_clicked(self):
        self.set_view_mode("icon")

    def list_mode_clicked(self):
        self.set_view_mode("list")

    def set_view_mode(self, view_mode):
        """아이콘과 리스트 모드로 변경하는 메소드"""
        if view_mode == "icon":
            mode = QtWidgets.QListView.IconMode
            stylesheet = """
                QListView {
                    padding: 2px;
                    background-color: #383838;
                }

                QListView::item {
                    border: 1px solid #518e55;
                    border-radius: 3px;
                    background-color: #3b5045;
                    margin: 2px;
                    color: white;
                }

                QListView::item::hover {
                    background-color: #4d8453;
                }"""
        else:
            mode = QtWidgets.QListView.ListMode
            stylesheet = "QListView {color: #cccccc;}"

        self.ui.list_view.setStyleSheet(stylesheet)
        self.ui.list_view.setViewMode(mode)

    def pop_list_view_menu(self, pos):
        """컨텍스트 메뉴를 보여주는 메소드"""
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.action_edit)
        # menu.addAction(self.action_delete)
        menu.setStyleSheet("QMenu::item {color: #cccccc;}\
                           QMenu::item:selected {background-color: #81CF3E; color: #404040;}")
        menu.exec_(self.ui.list_view.mapToGlobal(pos))

    def list_view_clicked(self, model_index):
        """태그를 클릭하면 clicked 시그널을 발생시키는 메소드"""
        if model_index.isValid():
            tag_model_index = self.tag_proxy_model.mapToSource(model_index)
            tag_item = self.tag_model.item(tag_model_index.row())

            document = self.tag_collection[tag_item]
            self.clicked.emit(document)

    def filter_line_edit_changed(self, text):
        """필터링 메소드"""
        reg_exp = QtCore.QRegExp(
            ustr(text).lower(), QtCore.Qt.CaseInsensitive, QtCore.QRegExp.FixedString)
        self.tag_proxy_model.setFilterRegExp(reg_exp)

    def update_list_view(self):
        """태그리스트를 갱신하는 메소드"""
        self.tag_collection = {}
        self.tag_dictionary = {}
        self.tag_model.clear()

        for document in Database.GetTagItems():
            item = HashableQStandardItem(document["name"])
            self.tag_model.appendRow(item)
            self.tag_collection[item] = document
            self.tag_dictionary[document["_id"]] = document["name"]

        self.ui.list_view.setModel(self.tag_proxy_model)
        self.ui.result_label.setText("{} results".format(self.tag_model.rowCount()))

    def update_label_clicked(self):
        self.update_list_view()
        # TODO: 태그 업데이트할 때 changed 시그널 발생이 필요한지?
        self.changed.emit(self.tag_dictionary)

    def add_btn_clicked(self):
        """태그 추가하는 메소드"""
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Tag", "Name:")
        if ok:
            tag_name = ustr(text)
            if tag_name == '':
                error_message("name is required!")
                return

            if self.tag_model.findItems(tag_name, QtCore.Qt.MatchFixedString):
                error_message("It already exists.")
                return

            try:
                Database.AddTagItem(tag_name)
            except:
                return

            self.update_list_view()
            self.changed.emit(self.tag_dictionary)

    def edit_btn_clicked(self):
        """태그 수정하는 메소드"""
        model_index = self.ui.list_view.currentIndex()
        if not model_index.isValid():
            return

        tag_model_index = self.tag_proxy_model.mapToSource(model_index)
        item = self.tag_model.item(tag_model_index.row())

        tag_name = ustr(item.text())
        text, ok = QtWidgets.QInputDialog.getText(self, "Edit Tag", "Name:", text=tag_name)
        if ok:
            new_tag_name = ustr(text)
            if new_tag_name == '':
                error_message("name is required!")
                return

            if (tag_name.lower() == new_tag_name.lower() or
                self.tag_model.findItems(new_tag_name, QtCore.Qt.MatchFixedString)):
                error_message("It already exists.")
                return

            try:
                document = Database.EditTagItem(new_tag_name, self.tag_collection[item]["_id"])
            except:
                return

            item.setText(document["name"])
            self.tag_collection[item] = document
            self.tag_dictionary[document["_id"]] = document["name"]

            self.tag_model.sort(0)

            self.changed.emit(self.tag_dictionary)

    def delete_btn_clicked(self):
        """태그 삭제하는 메소드"""
        model_index = self.ui.list_view.currentIndex()
        if not model_index.isValid():
            return

        tag_model_index = self.tag_proxy_model.mapToSource(model_index)
        item = self.tag_model.item(tag_model_index.row())

        tag_name = ustr(item.text())
        msg = QtWidgets.QMessageBox.question(
            self, "Delete Tag", u"Are you sure want to delete '{}'?".format(tag_name),
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        if msg != QtWidgets.QMessageBox.Ok:
            return

        # TODO: 아이템이 존재하는지 무결성 체크

        try:
            document = Database.DeleteTagItem(self.tag_collection[item]["_id"])
        except:
            return

        self.tag_model.takeRow(tag_model_index.row())
        del self.tag_collection[item]

        self.ui.result_label.setText("{} results".format(self.tag_model.rowCount()))

        # TODO: 카운트 갱신

def main():
    import sys
    images = [
        "/Users/rndvfx/synctest/tengyart-kSvpTrfhaiU-unsplash.jpg",
        "/Users/rndvfx/synctest/vadim-kaipov-f6jkAE1ZWuY-unsplash.jpg",
        "/Users/rndvfx/synctest/tim-hufner-nAMLTEerpWI-unsplash.jpg",
        "/Users/rndvfx/synctest/sam-moqadam-_kn16nC2vcw-unsplash.jpg",
        "/Users/rndvfx/synctest/girl-with-red-hat-r4A-lJTgXQg-unsplash.jpg"
    ]
    current_index = 0

    app = QtWidgets.QApplication(sys.argv)
    mainView = TagManager()
    mainView.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
