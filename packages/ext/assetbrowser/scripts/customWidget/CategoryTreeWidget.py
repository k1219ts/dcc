#coding=utf-8
from PySide2 import QtWidgets, QtGui, QtCore
from core import Database


class categoryTreeWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidget.__init__(self, parent)

        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.assetContext)
        self.Categorylist()
        # print(Database.GetCategoryList())

    def Categorylist(self):
        self.category_dict = Database.GetCategoryList()
        self.category_list = self.category_dict.keys()
        self.category_list.sort()
        self.category_list.remove('Default')
        self.category_list.insert(0, 'Default')
        self.category_list.remove('Texture')
        self.category_list.append('Texture')
        self.category_list.remove('HDRI')
        self.category_list.append('HDRI')
        for i in self.category_list:
            self.categoryItem = QtWidgets.QTreeWidgetItem([i])
            self.addTopLevelItem(self.categoryItem)
            self.categoryItem.setExpanded(True)
            subList =Database.GetCategoryList()[i]
            subList.sort()
            if 'Updated Source' in subList:
                subList.remove('Updated Source')
                subList.insert(0,'Updated Source')
                subList.remove('Trash')
                subList.append('Trash')
            if 'Updated Asset' in subList:
                subList.remove('Updated Asset')
                subList.insert(0,'Updated Asset')
                subList.remove('Trash')
                subList.append('Trash')

            for s in subList:
                # s= s.keys()
                sub = QtWidgets.QTreeWidgetItem([s])
                self.categoryItem.addChild(sub)

                # getSub = Database.gDB.category.find_one({'subCategory': s})
                # subID= getSub['_id']
                # getChild = Database.gDB.category.find({'childCategory':  {'$exists': True}})
                # for i in getChild:
                #     if i['subID'] == subID:
                #         item = QtWidgets.QTreeWidgetItem(sub,[i['childCategory']])
                #         self.categoryItem.addChild(item)


    def assetContext(self,point):
        checkSub= self.indexOfTopLevelItem(self.currentItem())
        current = self.currentItem()
        # print(checkSub)
        if self.itemAt(point) is None:
            # Qmenu
            menu = QtWidgets.QMenu()
            add_action = menu.addAction("Add Category")
            action = menu.exec_(self.mapToGlobal(point))

            if action == add_action:
                self.makeCategory()
            else:
                pass

        if checkSub == -1:#Sub Category
            currentSub = current.text(0)
            menu = QtWidgets.QMenu()

            if currentSub == 'Updated Source' or currentSub == 'Updated Asset' or currentSub == 'Trash':
                pass

            elif current.parent().text(0) == "Texture":
                # add_actionChild = menu.addAction("ADD Child Category")
                add_action = menu.addAction("Edit Sub Category")
                add_action2 = menu.addAction("Delete Sub Category")
                action = menu.exec_(self.mapToGlobal(point))
                category = current.parent().text(0)
                # if action == add_actionChild:
                #     self.makeChildCategory(category,currentSub)
                if action == add_action:
                    self.editSub()
                if action == add_action2:
                    self.delSub()
                else:
                    pass

            elif current.parent().parent():
                add_action = menu.addAction("Edit Child Category")
                add_action2 = menu.addAction("Delete Child Category")
                action = menu.exec_(self.mapToGlobal(point))
                if action == add_action:
                    self.editSub()
                if action == add_action2:
                    self.delChild()
                else:
                    pass

            else:
                add_action = menu.addAction("Edit Sub Category")
                add_action2 = menu.addAction("Delete Sub Category")
                action = menu.exec_(self.mapToGlobal(point))
                if action == add_action:
                    self.editSub()
                if action == add_action2:
                    self.delSub()
                else:
                    pass

        elif self.itemAt(point):
            index = self.indexAt(point)
            if not index.isValid():
                return
            menu = QtWidgets.QMenu()
            add_action = menu.addAction("Add Sub Category")
            action = menu.exec_(self.mapToGlobal(point))

            if current.text(0) == 'Default':
                pass
            elif action == add_action:
                self.makeSubCategory()
            else:
                pass

    def editSub(self):
        current = self.currentItem()
        parent = current.parent()
        currentSub = current.text(0)
        currentParent = current.parent().text(0)

        if current.parent().parent():
            text, ok = QtWidgets.QInputDialog.getText(self, ' Edit Child Category', 'ChildCategory Name')
            if ok:
                new = str(text)
                category = current.parent().parent().text(0)
                # Database.EditChildCategory(currentSub, new, category)
                parent.removeChild(current)
                child = QtWidgets.QTreeWidgetItem(parent, [new])
                self.addTopLevelItem(child)
        else:
            text, ok = QtWidgets.QInputDialog.getText(self, ' Edit Sub Category', 'SubCategory Name')
            if ok:
                new = str(text)
                Database.EditSubCategory(currentSub, new, currentParent)
                parent.removeChild(current)
                child = QtWidgets.QTreeWidgetItem(parent,[new])
                self.addTopLevelItem(child)


    def delSub(self):
        current = self.currentItem()
        currentSub = current.text(0)
        currentParent = current.parent().text(0)
        Database.DeleteSubCategory(currentSub,currentParent)
        current.parent().removeChild(current)


    def delChild(self):
        current = self.currentItem()
        sub = current.parent()
        currentChild = current.text(0)
        category = current.parent().parent().text(0)
        Database.DeleteChildCategory(currentChild,sub,category)
        sub.removeChild(current)


    def makeCategory(self):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'Category')
        if ok:
            new = str(text)
            QtWidgets.QTreeWidgetItem()

            categoryList=[]
            for i in self.category_list:
                i = i.lower()
                categoryList.append(i)

            if new.lower() in categoryList:
                self.ErrorMsg('It already exists.')
            else:
                cItem = QtWidgets.QTreeWidgetItem([new])
                self.addTopLevelItem(cItem)
                Database.AddCategory(new)

    def makeSubCategory(self):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'SubCategory Name')
        if ok:
            new = str(text)

            current = self.currentItem() # SubCategory
            childList =[]
            # print(current.childCount())
            for i in range(current.childCount()):
                name = current.child(i).text(0)
                childList.append(name)

            if new in childList:
                self.ErrorMsg('It already exists.')
            else:
                child = QtWidgets.QTreeWidgetItem(current,[new])
                self.addTopLevelItem(child)
                Database.AddSubCategory(mainCategory=current.text(0), subCategoryName=new)

    def makeChildCategory(self,category,currentSub):
        text, ok = QtWidgets.QInputDialog.getText(self, ' Text Input Dialog', 'ChildCategory Name')
        if ok:
            new = str(text)
            current = self.currentItem()
            childList =[]
            for i in range(current.childCount()):
                name = current.child(i).text(0)
                childList.append(name)
            if new in childList:
                self.ErrorMsg('It already exists.')
            else:
                child = QtWidgets.QTreeWidgetItem(current,[new])
                self.addTopLevelItem(child)
                Database.AddchildCategory(mainCategory=category, subCategory=current.text(0), childCategory=new)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        category_target = self.itemAt(event.pos())
        # checkSub = self.indexOfTopLevelItem(category_target)

        if category_target is None:
            pass

        elif category_target.parent().parent():  # child
            source_items = event.source().selectedItems()
            sub=category_target.parent()
            subCategory= sub.text(0)

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
                    source_name = item.text()
                    sourceDB = []
                    for i in Database.gDB.source.find({"name": source_name}):
                        sourceDB.append(i)
                    objId = sourceDB[0]["_id"]
                    Database.MoveChildCategory(objId, categoryName, subCategory, child)
                    event.source().takeItem(event.source().row(item))

        elif category_target.parent(): #sub_category
            source_items = event.source().selectedItems()
            targetCategory = category_target.parent().text(0)
            targetSub = category_target.text(0)
            current = self.currentItem()
            currentName = current.text(0)
            currentParent = current.parent().text(0)

            if currentParent != 'Texture' and targetCategory != 'Texture': #USD
                for item in source_items:
                    source_name = item.text()
                    sourceDB = []
                    for i in Database.gDB.item.find({"name": source_name}):
                        sourceDB.append(i)
                    objId = sourceDB[0]["_id"]
                    Database.MoveCategoryTest(objId, targetCategory, targetSub)
                    event.source().takeItem(event.source().row(item))

            elif currentParent == 'Texture' and targetCategory == 'Texture': #texture source move to 'Texture' Category
                for item in source_items:
                    source_name = item.text()
                    sourceDB = []
                    for i in Database.gDB.source.find({"name": source_name}):
                        sourceDB.append(i)
                    objId = sourceDB[0]["_id"]
                    Database.MoveCategoryTest(objId, targetCategory, targetSub)
                    event.source().takeItem(event.source().row(item))

            # else:
            #     if currentParent == 'Texture':
            #         self.ErrorMsg('This is a texture source.')
            #
            #     else:
            #         self.ErrorMsg('This is a USD Asset.')




        elif category_target.parent() is None: #main_category
            pass
        else:
            pass


    def ErrorMsg(self, msgText, title = "Error", button = QtWidgets.QMessageBox.Ok):
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowTitle(title)
        msgBox.setText(msgText)
        msgBox.setStandardButtons(button)
        return msgBox.exec_()
