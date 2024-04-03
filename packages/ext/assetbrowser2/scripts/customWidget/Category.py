# -*- coding: utf-8 -*-
from pymodule.Qt import QtCore
from pymodule.Qt import QtWidgets

from core import Database
from libs.customError import CustomError
from libs.utils import error_message

class CategoryTreeWidget(QtWidgets.QTreeWidget):

    changed = QtCore.Signal()
    updated = QtCore.Signal()

    def __init__(self, parent=None):
        QtWidgets.QTreeWidget.__init__(self, parent)

        # instance variables
        self.advanced_mode = True
        self.category_list = []

        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.assetContext)

        self.menu = QtWidgets.QMenu(self)
        self.menu.setStyleSheet("\
            QMenu::item {color: #CCCCCC;}\
            QMenu::item:selected {background-color: #81CF3E; color: #404040;}")
        self.add_sub_category_action = QtWidgets.QAction("Add Sub Category", self)
        self.add_sub_category_action.triggered.connect(self.makeSubCategory)
        self.edit_sub_category_action = QtWidgets.QAction("Edit Sub Category", self)
        self.edit_sub_category_action.triggered.connect(self.editSub)
        self.delete_sub_category_action = QtWidgets.QAction("Delete Sub Category", self)
        self.delete_sub_category_action.triggered.connect(self.delSub)
        self.edit_child_category_action = QtWidgets.QAction("Edit Child Category", self)
        self.edit_child_category_action.triggered.connect(self.editSub)
        self.delete_category_action = QtWidgets.QAction("Delete Category", self)
        self.delete_category_action.triggered.connect(self.delete_category)

        self.reload_list()

    def is_feature_enabled(self):
        return self.advanced_mode

    def set_advanced_mode(self, mode):
        self.advanced_mode = mode

    def reload_list(self):
        self.clear()

        category_dict = Database.GetCategoryList()
        if not category_dict:
            return

        self.category_list = sorted(category_dict.keys())
        self.category_list.remove('Default')
        self.category_list.insert(0, 'Default')
        self.category_list.remove('Texture')
        self.category_list.append('Texture')
        for i in self.category_list:
            categoryItem = QtWidgets.QTreeWidgetItem([i])
            self.addTopLevelItem(categoryItem)
            categoryItem.setExpanded(True)
            subList = sorted(Database.GetCategoryList()[i])
            if 'Updated Source' in subList:
                subList.remove('Updated Source')
                subList.insert(0, 'Updated Source')
                subList.remove('Trash')
                subList.append('Trash')
            if 'Updated Asset' in subList:
                subList.remove('Updated Asset')
                subList.insert(0, 'Updated Asset')
                subList.remove('Trash')
                subList.append('Trash')

            for s in subList:
                # s= s.keys()
                sub = QtWidgets.QTreeWidgetItem([s])
                categoryItem.addChild(sub)

                # getSub = Database.gDB.category.find_one({'subCategory': s})
                # subID= getSub['_id']
                # getChild = Database.gDB.category.find({'childCategory':  {'$exists': True}})
                # for i in getChild:
                #     if i['subID'] == subID:
                #         item = QtWidgets.QTreeWidgetItem(sub,[i['childCategory']])
                #         categoryItem.addChild(item)


    def assetContext(self, point):
        self.menu.clear()

        item = self.indexAt(point)
        if not item.isValid():
            return

        if not self.is_feature_enabled():
            self.menu.exec_(self.mapToGlobal(point))
            return

        checkSub = self.indexOfTopLevelItem(self.currentItem())
        current = self.currentItem()
        if checkSub == -1: # Sub Category
            currentSub = current.text(0)
            if currentSub in ["Updated Source", "Updated Asset", "Trash"]:
                pass

            elif current.parent().text(0) == "Texture":
                # add_actionChild = menu.addAction("ADD Child Category")
                self.menu.addAction(self.edit_sub_category_action)
                self.menu.addAction(self.delete_sub_category_action)
                action = self.menu.exec_(self.mapToGlobal(point))
                # category = current.parent().text(0)
                # if action == add_actionChild:
                #     self.makeChildCategory(category,currentSub)

            elif current.parent().parent():
                self.menu.addAction(self.edit_child_category_action)
                self.menu.addAction(self.delete_sub_category_action)
                self.menu.exec_(self.mapToGlobal(point))

            else: # sub_category
                self.menu.addAction(self.edit_sub_category_action)
                self.menu.addAction(self.delete_sub_category_action)
                self.menu.exec_(self.mapToGlobal(point))

        elif self.itemAt(point): # Category
            self.menu.addAction(self.add_sub_category_action)
            if current.text(0) not in ["Texture", "Default"]:
                self.menu.addAction(self.delete_category_action)
            self.menu.exec_(self.mapToGlobal(point))

        else:
            pass

    def editSub(self):
        current = self.currentItem()
        parent = current.parent()
        currentSub = current.text(0)
        currentParent = current.parent().text(0)

        if current.parent().parent():
            text, ok = QtWidgets.QInputDialog.getText(self, ' Edit Child Category',
                                                      'ChildCategory Name')
            if ok and (str(text) != currentSub):
                new = str(text)
                if new == '':
                    error_message('name is required!')
                    return

                category = current.parent().parent().text(0)
                # Database.EditChildCategory(currentSub, new, category)
                parent.removeChild(current)
                child = QtWidgets.QTreeWidgetItem(parent, [new])
                self.addTopLevelItem(child)
        else:
            text, ok = QtWidgets.QInputDialog.getText(self, ' Edit Sub Category',
                                                      'SubCategory Name', text=currentSub)
            if ok and (str(text) != currentSub):
                new = str(text)
                if new == '':
                    error_message('name is required!')
                    return

                Database.EditSubCategory(currentSub, new, currentParent)
                parent.removeChild(current)
                child = QtWidgets.QTreeWidgetItem(parent, [new])
                self.addTopLevelItem(child)

                for row in range(parent.childCount()):
                    child_item = parent.child(row)
                    if child_item.text(0) == new:
                        self.setCurrentItem(child_item)
                        break

                self.changed.emit()

    def delSub(self):
        msg = QtWidgets.QMessageBox.question(
            self, "Delete", u"Are you sure want to delete items?",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        if msg != QtWidgets.QMessageBox.Ok:
            return

        current = self.currentItem()
        parent = current.parent()
        currentSub = current.text(0)
        currentParent = current.parent().text(0)
        Database.DeleteSubCategory(currentSub, currentParent)
        current.parent().removeChild(current)

        self.setCurrentItem(parent)
        self.changed.emit()

    def delChild(self):
        current = self.currentItem()
        sub = current.parent()
        currentChild = current.text(0)
        category = current.parent().parent().text(0)
        Database.DeleteChildCategory(currentChild, sub, category)
        sub.removeChild(current)

    def delete_category(self):
        current = self.currentItem()
        category_index = self.indexOfTopLevelItem(current)
        category = current.text(0)
        try:
            Database.DeleteCategory(category)
        except CustomError as error:
            error_message(str(error))
            return

        self.updated.emit()

    def makeCategory(self):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'Category')
        if ok:
            new = str(text)
            if new == '':
                error_message('name is required!')
                return

            categoryList = []
            for i in self.category_list:
                i = i.lower()
                categoryList.append(i)

            if new.lower() in categoryList:
                error_message('It already exists.')
                return

            try:
                Database.AddCategory(new)
                cItem = QtWidgets.QTreeWidgetItem([new])
                self.addTopLevelItem(cItem)
            except CustomError as error:
                error_message(str(error))

            for row in range(self.topLevelItemCount()):
                item = self.topLevelItem(row)
                if item.text(0) == new:
                    self.setCurrentItem(item)
                    self.changed.emit()
                    break

    def makeSubCategory(self):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'SubCategory Name')
        if ok:
            new = str(text)
            if new == '':
                error_message('name is required!')
                return

            current = self.currentItem() # SubCategory
            childList = []
            # print current.childCount()
            for i in range(current.childCount()):
                name = current.child(i).text(0)
                childList.append(name)

            if new in childList:
                error_message('It already exists.')
                return

            try:
                Database.AddSubCategory(mainCategory=current.text(0), subCategoryName=new)
                child = QtWidgets.QTreeWidgetItem(current, [new])
                self.addTopLevelItem(child)
            except CustomError as error:
                error_message(str(error))
                return

            self.expandItem(current)

    def makeChildCategory(self, category, currentSub):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'ChildCategory Name')
        if ok:
            new = str(text)
            if new == '':
                error_message('name is required!')
                return

            current = self.currentItem()
            childList = []
            for i in range(current.childCount()):
                name = current.child(i).text(0)
                childList.append(name)
            if new in childList:
                error_message('It already exists.')
            else:
                child = QtWidgets.QTreeWidgetItem(current, [new])
                self.addTopLevelItem(child)
                Database.AddchildCategory(mainCategory=category, subCategory=current.text(0), childCategory=new)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        className = event.source().metaObject().className()
        if className == 'BookmarkView':
            return

        category_target = self.itemAt(event.pos())
        # checkSub = self.indexOfTopLevelItem(category_target)

        if category_target is None:
            pass

        elif category_target.parent().parent():  # child
            source_items = event.source().selectedItems()
            sub = category_target.parent()
            subCategory = sub.text(0)
            categoryName = sub.parent().text(0)
            child = category_target.text(0)
            current = self.currentItem()
            currentParent = current.parent().text(0)

            # if currentParent != 'Texture' and targetCategory != 'Texture':  # USD
            #     for item in source_items:
            #         source_name = item.text()
            #         sourceDB = []
            #         for i in Database.gDB.item.find({"name": source_name}):
            #             sourceDB.append(i)
            #         objId = sourceDB[0]["_id"]
            #         Database.MoveCategoryTest(objId, targetCategory, targetSub)
            #         event.source().takeItem(event.source().row(item))
            if currentParent == 'Texture' and categoryName == 'Texture':  # texture source move to 'Texture' Category
                for item in source_items:
                    document = event.source().document(item)
                    Database.MoveChildCategory(document.object_id, categoryName, subCategory, child)
                    event.source().takeItem(event.source().row(item))

        elif category_target.parent(): #sub_category
            targetCategory = category_target.parent().text(0)
            targetSub = category_target.text(0)
            current = self.currentItem()
            currentParent = current.parent().text(0) # TODO: AttributError: NoneType 에러 발생

            # https://stackoverflow.com/questions/11246022/how-to-get-qstring-from-qlistview-selected-item-in-qt
            # QVector<QItemSelectionRange> ranges = ui.listView->selectionModel()->selection().toVector();
            # foreach (const QItemSelectionRange& range, ranges) {
            #     ui.listView->model()->removeRows(range.top(), range.height());
            # }

            ranges = event.source().selectionModel().selection()
            if currentParent != 'Texture' and targetCategory != 'Texture': #USD
                for r in ranges:
                    for model_index in r.indexes():
                        document = event.source().find_document(model_index)
                        Database.MoveCategoryTest(document.object_id, targetCategory, targetSub)
                    event.source().model().removeRows(r.top(), r.height())
                self.changed.emit()
                event.accept()

            elif currentParent == 'Texture' and targetCategory == 'Texture': #texture source move to 'Texture' Category
                for r in ranges:
                    for model_index in r.indexes():
                        document = event.source().find_document(model_index)
                        Database.MoveCategoryTest(document.object_id, targetCategory, targetSub)
                    event.source().model().removeRows(r.top(), r.height())
                self.changed.emit()
                event.accept()

            # else:
            #     if currentParent == 'Texture':
            #         error_message('This is a texture source.')
            #
            #     else:
            #         error_message('This is a USD Asset.')




        elif category_target.parent() is None: #main_category
            pass
        else:
            pass

    # def mousePressEvent(self, event):
    #     item = self.indexAt(event.pos())
    #     if not item.isValid():
    #         self.clearSelection()
    #     QtWidgets.QTreeWidget.mousePressEvent(self, event)
