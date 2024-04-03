#-*- coding: utf-8 -*-
import copy
import datetime
import getpass
import os
import sys

from pymodule.Qt import QtCompat
from pymodule.Qt import QtCore
from pymodule.Qt import QtGui
from pymodule.Qt import QtWidgets

from core import Database
from widgets.ui_itemform import Ui_ItemForm
from libs.customError import CustomError
from libs.ustr import ustr
from libs.utils import error_message

# TODO: Add / Edit 분리할 것
class ItemForm(QtWidgets.QDialog):

    changed = QtCore.Signal()
    tag_changed = QtCore.Signal()

    def __init__(self, parent, document=None):
        super(ItemForm, self).__init__(parent)
        # ui_path = os.path.dirname(os.path.abspath(__file__))
        # self.ui = QtCompat.loadUi(os.path.join(ui_path, "ui", "editform.ui"), self)
        self.ui = Ui_ItemForm()
        self.ui.setupUi(self)

        if document is None: # add item
            window_title = "Add Item"
            self.document_parent = None
            self.document = None
            self.is_asset = None
            self.category = "Default" # 한번만 사용
            self.tags = []
        else: # edit item
            window_title = "Edit Item"
            self.document_parent = document
            self.document = copy.deepcopy(document)
            self.is_asset = True
            if document.category == "Texture":
                self.is_asset = False
            self.category = self.document.category # 한번만 사용
            self.tags = self.document.tags

        self.categories = {}
        self.last_open_dir = None
        self.is_tag_changed = False

        self.setModal(True)
        self.setWindowTitle(window_title)
        self.ui.status_combo.addItems(["", "Publish", "Delete"])
        self.ui.splitter.setStretchFactor(0, 3)
        self.ui.splitter.setStretchFactor(1, 1)

        # move a window
        try:
            dialog_center = self.mapToGlobal(self.rect().center())
            parent_window_center = self.parent().window().mapToGlobal(
                self.parent().window().rect().center()
            )
            self.move(parent_window_center-dialog_center)
        except:
            pass

        self.ui.files_preview_btn.clicked.connect(self.files_preview_btn_clicked)
        self.ui.files_path_btn.clicked.connect(self.files_path_btn_clicked)
        self.ui.storage_file_path_btn.clicked.connect(self.storage_file_path_btn_clicked)
        self.ui.set_preview_btn.clicked.connect(self.set_preview)
        self.ui.remove_image_btn.clicked.connect(self.remove_list)
        self.ui.ok_btn.clicked.connect(self.submit)
        self.ui.cancel_btn.clicked.connect(self.close)

        self.ui.tag_manager_widget.clicked.connect(self.tag_manager_widget_clicked)
        self.ui.tag_manager_widget.changed.connect(self.update_tag_list_widget)
        self.ui.tag_list_widget.itemClicked.connect(self.tag_list_widget_clicked)

        self.ui.category_combo.currentIndexChanged[str].connect(self.category_changed)
        self.ui.name_edit.textChanged.connect(self.name_text_changed)

    def name_text_changed(self, text):
        self.ui.name_edit.backgroundRole()
        if not text:
            self.ui.name_edit.setStyleSheet("color: #ffffff; background-color: #412529;")
        else:
            self.ui.name_edit.setStyleSheet("color: #cccccc; background-color: #383838;")

    def open_file(self, line_edit, file_path, custom_filters=None):
        path = os.path.dirname(ustr(file_path)) if file_path else '.'
        if custom_filters:
            filters = custom_filters
        else:
            formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QtGui.QImageReader.supportedImageFormats()]
            filters = "Image files (%s)" % ' '.join(formats)
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose Image file', path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
                if not filename:
                    return
            line_edit.setText(filename)

    def open_dir_dialog(self, line_edit, dir_path=None, silent=False):
        default_open_dir_path = dir_path if dir_path else '.'
        if self.last_open_dir and os.path.exists(self.last_open_dir):
            default_open_dir_path = self.last_open_dir
        # else:
        #     default_open_dir_path = os.path.dirname(file_path) if file_path else '.'
        if silent != True:
            target_dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                'Open Directory', default_open_dir_path,
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks)
            if target_dir_path:
                line_edit.setText(target_dir_path)
        else:
            target_dir_path = default_open_dir_path
        self.last_open_dir = target_dir_path

    def files_preview_btn_clicked(self):
        self.open_file(self.ui.files_preview_edit, self.ui.files_preview_edit.text())

    def files_path_btn_clicked(self):
        if ustr(self.ui.category_combo.currentText()) == "Texture":
            self.open_dir_dialog(self.ui.files_usdfile_edit, self.ui.files_usdfile_edit.text())
        else:
            self.open_file(self.ui.files_usdfile_edit, self.ui.files_usdfile_edit.text(),
                           custom_filters="USD file (*.usd)")

    def storage_file_path_btn_clicked(self):
        self.open_dir_dialog(self.ui.storage_file_path_edit, self.ui.storage_file_path_edit.text())

    def category_changed(self, category):
        self.ui.category_combo.currentIndexChanged[str].disconnect(self.category_changed)
        current_text = self.ui.sub_category_combo.currentText()
        self.ui.sub_category_combo.clear()
        find_text = False
        unknown_index = -1
        for row, sub_category in enumerate(sorted(self.categories[category])):
            self.ui.sub_category_combo.addItem(sub_category)
            if current_text == sub_category:
                self.ui.sub_category_combo.setCurrentIndex(row)
                find_text = True
            if "unknown" == sub_category:
                unknown_index = row

        if not find_text:
            if unknown_index != -1:
                self.ui.sub_category_combo.setCurrentIndex(unknown_index)

        if category == "Texture":
            self.ui.files_usdfile_label.setText("files.filePath")
        else:
            self.ui.files_usdfile_label.setText("files.usdfile")

        self.ui.category_combo.currentIndexChanged[str].connect(self.category_changed)

    def update_form_for_add(self, category, sub_category):
        """아이템 추가할 때 사용하는 메소드"""
        categories = Database.GetCategoryList()
        self.categories = categories

        self.ui.category_combo.currentIndexChanged[str].disconnect(self.category_changed)
        for row, c_category in enumerate(sorted(categories.keys())):
            self.ui.category_combo.addItem(c_category)
            if category == c_category:
                self.ui.category_combo.setCurrentIndex(row)
        self.ui.category_combo.currentIndexChanged[str].connect(self.category_changed)

        for row, c_sub_category in enumerate(sorted(categories[category])):
            self.ui.sub_category_combo.addItem(c_sub_category)
            if sub_category == c_sub_category:
                self.ui.sub_category_combo.setCurrentIndex(row)

    def update_form(self):
        """아이템 수정할 때 사용하는 메소드"""
        categories = Database.GetCategoryList()
        if self.is_asset:
            del categories["Texture"]
        else:
            categories = {"Texture": categories["Texture"]}
        self.categories = categories

        self.ui.category_combo.currentIndexChanged[str].disconnect(self.category_changed)
        for row, category in enumerate(sorted(categories.keys())):
            self.ui.category_combo.addItem(category)
            # set index
            if self.document.category == category:
                self.ui.category_combo.setCurrentIndex(row)
        self.ui.category_combo.currentIndexChanged[str].connect(self.category_changed)

        for row, sub_category in enumerate(sorted(categories[self.document.category])):
            self.ui.sub_category_combo.addItem(sub_category)
            if self.document.sub_category == sub_category:
                self.ui.sub_category_combo.setCurrentIndex(row)

        self.ui.name_edit.setText(self.document.name)

        preview = self.document.preview_path
        if preview:
            self.ui.files_preview_edit.setText(preview)

        if self.document.category == "Texture":
            self.ui.files_usdfile_label.setText("files.filePath")
            usdfile_path = self.document.file_path
        else:
            self.ui.files_usdfile_label.setText("files.usdfile")
            usdfile_path = self.document.usdfile
        self.ui.files_usdfile_edit.setText(usdfile_path)

        storage_file_path = self.document.storage_file_path
        if storage_file_path:
            self.ui.storage_file_path_edit.setText(storage_file_path)

        storage_file_size = self.document.storage_file_size
        if storage_file_size:
            self.ui.storage_file_size_edit.setText(storage_file_size)

        status = self.document.status
        if status:
            status_index = self.ui.status_combo.findText(status)
            if status_index:
                self.ui.status_combo.setCurrentIndex(status_index)

        tag = self.document.tag
        if tag:
            self.ui.tag_edit.setText(u','.join(tag))

        # TODO: objectID 가지고 올 수 있도록 처리
        for tag_object in self.document.tag_objects:
            item = QtWidgets.QListWidgetItem()
            item.setText(tag_object["name"])
            item.setData(QtCore.Qt.UserRole, tag_object["_id"])
            self.ui.tag_list_widget.addItem(item)

        comment = self.document.comment
        if comment:
            self.ui.comment_edit.setPlainText(comment)

        images = self.document.images
        if images:
            for image in images:
                self.ui.image_list.addItem(image)

    def submit(self):
        category = self.ui.category_combo.currentText()
        sub_category = self.ui.sub_category_combo.currentText()

        datas = {}
        datas["category"] = category
        datas["subCategory"] = sub_category
        datas["name"] = self.ui.name_edit.text()

        if not self.ui.name_edit.text():
            error_message("name is required!")
            return

        files = {"preview": self.ui.files_preview_edit.text()}
        usdfile_path = self.ui.files_usdfile_edit.text()
        if category == "Texture":
            files["filePath"] = usdfile_path
        else:
            files["usdfile"] = usdfile_path
        datas["files"] = files
        datas["tag"] = ustr(self.ui.tag_edit.text()).split(',')
        datas["comment"] = self.ui.comment_edit.toPlainText()

        if Database.gDBNAME != "ASSETLIB":
            datas["storageFilePath"] = self.ui.storage_file_path_edit.text()
            datas["storageFileSize"] = self.ui.storage_file_size_edit.text()
            datas["status"] = self.ui.status_combo.currentText()

            images = []
            for row in range(self.ui.image_list.count()):
                item = self.ui.image_list.item(row)
                images.append(item.text())
            datas["images"] = images
            datas["tags"] = self.tags

        if self.document is None: # add item
            datas["reply"] = [
                {
                    'user':getpass.getuser(),
                    'comment': 'add item',
                    'time':datetime.datetime.now().isoformat()
                }
            ]

            try:
                Database.AbstractAddItem(category, datas)
            except CustomError as error:
                error_message(str(error))
                return

            self.is_tag_changed = True # 리스트 갱신하기 위해 강제 발생

        else: # edit item

            # 카테고리의 드래그앤드롭 하는 방식처럼 변경할 것 - 이동 후 아이템만 삭제
            # 카테고리 또는 서브카테고리가 변경이 되었을 경우 이벤트 발생
            if category != self.document.category or sub_category != self.document.sub_category:
                self.is_tag_changed = True

            reply = [{'user':getpass.getuser(),
                     'comment': 'edit item',
                     'time':datetime.datetime.now().isoformat()}]

            document = Database.EditItem(category, self.document.object_id, datas, reply)
            self.document_parent.set_item(document)

            if not self.is_tag_changed:
                self.changed.emit()

        self.close()

    def set_preview(self):
        items = self.ui.image_list.selectedItems()
        if items and len(items) == 1:
            for item in items:
                row = self.ui.image_list.row(item)
                self.ui.files_preview_edit.setText(item.text())
                self.ui.image_list.takeItem(row)

    def remove_list(self):
        items = self.ui.image_list.selectedItems()
        for item in items:
            row = self.ui.image_list.row(item)
            self.ui.image_list.takeItem(row)

    def tag_manager_widget_clicked(self, document):
        if document["_id"] in self.tags:
            error_message("It already exists.")
        else:
            try:
                self.document.add_tag_object(document)
            except:
                self.tags.append(document["_id"])
            item = QtWidgets.QListWidgetItem()
            item.setText(document["name"])
            item.setData(QtCore.Qt.UserRole, document["_id"])
            self.ui.tag_list_widget.addItem(item)

    def tag_list_widget_clicked(self, item):
        object_id = item.data(QtCore.Qt.UserRole)
        try:
            self.document.remove_tag_object(object_id)
        except:
            del self.tags[self.tags.index(object_id)]
        row = self.ui.tag_list_widget.row(item)
        self.ui.tag_list_widget.takeItem(row)

    def update_tag_list_widget(self, tag_dictionary):
        for row in range(self.ui.tag_list_widget.count()):
            item = self.ui.tag_list_widget.item(row)
            tag_name = ustr(item.text())
            tag_id = item.data(QtCore.Qt.UserRole)

            new_tag_name = tag_dictionary[tag_id]
            if tag_name != new_tag_name:
                item.setText(new_tag_name)
                print("{} -> {}".format(tag_name, new_tag_name))

        self.is_tag_changed = True

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super(ItemForm, self).keyPressEvent(event)

    def closeEvent(self, event):
        if self.is_tag_changed:
            self.tag_changed.emit()

        super(ItemForm, self).closeEvent(event)

if __name__ == "__main__":
    # TODO: 정리
    from bson.objectid import ObjectId
    from models.asset import Asset
    from models.texture import Texture
    Database.set_database("ASSETLIB2")

    item = Database.gDB.item.find_one({"tags": {"$exists": True, "$ne": []}})
    document = Asset(item)

    # item = Database.gDB.source.find_one()
    # document = Texture(item)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("\
        QDialog, QListWidget, QListView, QPlainTextEdit, QPushButton, QToolButton, QLineEdit {\
            color: #cccccc; background-color: #383838; }")
    dialog = ItemForm(parent=None, document=document)
    # dialog.update_form_for_add("Default", "unknown")
    dialog.update_form()
    dialog.show()
    sys.exit(app.exec_())
