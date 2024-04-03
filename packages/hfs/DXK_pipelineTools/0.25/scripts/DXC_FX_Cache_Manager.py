# -*- coding: utf-8 -*-

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import hou, datetime, os, shutil, sys
from multiprocessing.pool import ThreadPool
import time

class HelloWindow(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent, Qt.WindowStaysOnTopHint)
        
        self._set_style()
        self.populate_tree()

    def refreshTree(self):
        self.tree_widget.clear() # clean so that deleted folder would not show up
        self.populate_tree()

    def populate_tree(self):
        sTime = time.time()
        current_directory = self.seq()
        pool = ThreadPool()

        pool.map(self.add_directory_to_tree_parallel, [(current_directory, self.tree_widget, current_directory)])
        pool.close()
        pool.join()

        eTime = time.time()
        elapsed = eTime - sTime
        
        try:
            print("Time spent :" + "%.3f"%round(elapsed,3) + " seconds")
        except:
            pass    

        self.searchMatchingVersion()
        
    def add_directory_to_tree_parallel(self, args):
        directory, parent_item, folder_path = args
        self.add_directory_to_tree(directory, parent_item, folder_path)

    def searchMatchingVersion(self):
        fileIO_dict = []
        arr= []
        listarr = []

        list = hou.node("/obj").allSubChildren()
        try:
            for node in list:
                if "FileIO" in node.type().name():
                    listarr.append(node)
        except:
            pass

        merged = tuple(listarr)

        for fileio in merged:
            fxGrp = fileio.parm("FX_GROUP").evalAsString()
            takev = fileio.parm("INPUT_TAKE").evalAsString()
            datatyp = fileio.parm("DATA_TYPE").evalAsString()
            elenm = datatyp + "_" + fileio.parm("ELEMENT_NAME").eval().lower()
            elever = fileio.parm("ELEMENT_VERSION").eval()
            entry_dict = {
                "fxGrp" : fxGrp,
                "takev" : takev,
                "datatyp" : datatyp,
                "elenm" : elenm,
                "elever" : elever,
            }
            fileIO_dict.append(entry_dict)

        for entry in fileIO_dict:
            #print(entry)
            self.search_in_tree(self.tree_widget.invisibleRootItem(),entry)
        
        nameList,fxGrp = self.getCacheName(self.tree_widget.invisibleRootItem(),fileIO_dict)

        #self.searchDeadCache(self.tree_widget.invisibleRootItem(),nameList,fxGrp)
        #print(nameList)

    def getCacheName(self,item, entry):
        nameList = []
        fxGrp = []
        for list in entry:
            fxGrp.append(list["fxGrp"])
            nameList.append(list["elenm"])

        #print(nameList)
        return nameList,fxGrp

    def searchDeadCache(self, item,nameList,fxGrp):
        for i in range(item.childCount()):
            child = item.child(i)
            try:
                if child.text(0) not in nameList:
                    #child.parent().setForeground(0, QBrush(QColor(215,110,110)))
                    pass
            except:
                pass

            self.searchDeadCache(child, nameList,fxGrp)


    def search_in_tree(self, item, entry):

        for i in range(item.childCount()):
            child = item.child(i)

            try:
                elenm = child.parent().text(0)
                #print("Parent : ", elenm)
            except AttributeError as e:
                print(e)

            try:
                takev = child.parent().parent().text(0)
                #print("Grandparent : ",takev)
            except AttributeError as e:
                print(e)

            elever = child.text(0)

            try:
                if entry["takev"] == takev and entry["elenm"] == elenm and entry["elever"] == elever:
                    child.setForeground(0, QBrush(QColor(255,165,0)))
            except:
                pass


            self.search_in_tree(child, entry)

    def add_directory_to_tree(self, directory, parent_item, folder_path, depth=0):       

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            item_name = os.path.basename(item_path)
            
            if os.path.isdir(item_path):
                size = self.get_directory_size(item_path)
                date = datetime.datetime.fromtimestamp(os.path.getmtime(item_path)).strftime('%m-%d  %H:%M')

                item_item = QTreeWidgetItem(parent_item, [item_name, self.format_size(size), date])
                item_item.setExpanded(True)
                item_item.setData(0, Qt.UserRole, os.path.join(folder_path, item_name))  
                self.add_directory_to_tree(item_path, item_item, folder_path, depth + 1)  # Pass folder_path

    def get_directory_size(self, path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def format_size(self, size):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return "%3.1f %s" % (size, unit)
            size /= 1024.0

    def confirm_delete_selected_folder(self):
        selected_items = self.tree_widget.selectedItems()

        if not selected_items:
            warnin_dialog = QMessageBox()
            warnin_dialog.setIcon(QMessageBox.Warning)
            warnin_dialog.setText("You have not selected any folder")
            result = warnin_dialog.exec_()
            #print("Selected Nothing")
            return
        else :
            message = "Are you sure you want to delete the selected folders?"
            confirm_dialog = QMessageBox()
            confirm_dialog.setIcon(QMessageBox.Warning)
            confirm_dialog.setWindowTitle("Confirm Deletion")
            confirm_dialog.setText(message)
            confirm_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            curs=QCursor.pos()
            confirm_dialog.move(curs.x(),curs.y())
            result = confirm_dialog.exec_()

            if result == QMessageBox.Yes:
                for item in selected_items:
                    self.delCache(item)
            else:
                pass
        
    def delCache(self, item):
        path = []
        check = item
        while item is not None:
            path.insert(0,item.text(0))
            item = item.parent()
        #return 
        cachepath = '/'.join(path)
        abspath = self.seq() + '/' + cachepath
        
        if os.path.exists(abspath) and (check.childCount() == 0):
            try:
                print("Path to remove : " + abspath)
                print("Children : " + str(check.childCount()))
            except:
                pass

            check.setForeground(0, QBrush(QColor(111,111,111)))

            #### Uncomment the following line to actually remove the folder
            shutil.rmtree(abspath)
            self.tree_widget.clearSelection()
            self.tree_widget.takeTopLevelItem(self.tree_widget.indexOfTopLevelItem(check))
            
        elif check.childCount() != 0 :
            self.foolProof()
        
    def foolProof(self):
        message = "Subdirectory Found. This is Discouraged. Aborting job."
        confirm_dialog = QMessageBox()
        confirm_dialog.setIcon(QMessageBox.Critical)
        confirm_dialog.setWindowTitle("Warning")
        confirm_dialog.setText(message)
        confirm_dialog.setStandardButtons(QMessageBox.Ok)
        curs=QCursor.pos()
        confirm_dialog.move(curs.x(),curs.y())
        result = confirm_dialog.exec_()

    def fail_finding_directory(self):
        message = "INVALID FILE LOCATION !"
        
        confirm_dialog = QMessageBox()
        confirm_dialog.setIcon(QMessageBox.Critical)
        confirm_dialog.setWindowTitle("Warning")
        confirm_dialog.setText(message)
        confirm_dialog.setStandardButtons(QMessageBox.Ok)
        curs=QCursor.pos()
        confirm_dialog.move(curs.x(),curs.y())
        result = confirm_dialog.exec_()
        sys.exit()

    def seq(self):

        hipname_ = "/show/" + hou.hipFile.name().split("/show/")[-1]
        split_ = hipname_.split("/")
        path=0

        if "_fx_" in split_[-1]:
            path = "/fx_cache/"+split_[2]+"/"+split_[6]+"/"+split_[7]+"/dev/"+"fx00"
        elif "_fx02_" in split_[-1]:
            path = "/fx_cache/"+split_[2]+"/"+split_[6]+"/"+split_[7]+"/dev/"+"fx02"
        elif "_fx03_" in split_[-1]:
            path = "/fx_cache/"+split_[2]+"/"+split_[6]+"/"+split_[7]+"/dev/"+"fx03"
        elif "_fx04_" in split_[-1]:
            path = "/fx_cache/"+split_[2]+"/"+split_[6]+"/"+split_[7]+"/dev/"+"fx04"
        elif "_fx05_" in split_[-1]:
            path = "/fx_cache/"+split_[2]+"/"+split_[6]+"/"+split_[7]+"/dev/"+"fx05"        
  
        if path != 0:
            return path
        else :
            self.fail_finding_directory()

    def _set_style(self):
        self.setWindowTitle("FX Cache Manager || by PFX yongjun.cho")
        self.setMinimumWidth(950)
        self.setMinimumHeight(700)
        self.setMaximumWidth(1400)
        self.setMaximumHeight(900)
        layout = QVBoxLayout(self)
        button_layout=QHBoxLayout()
        welcome_label = QLabel("Welcome to FX Cache Manager, " + str(os.environ.get('USERNAME')))
        welcome_label.setAlignment(Qt.AlignLeft)
        welcome_label.setStyleSheet("font-size: 17px; font-weight: bold; margin-bottom: 1px; color: rgb(133,133,133)")
        layout.addWidget(welcome_label)
        welcome_label2 = QLabel("Target : "+self.seq())
        welcome_label2.setAlignment(Qt.AlignLeft)
        welcome_label2.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 3px;")
        layout.addWidget(welcome_label2)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setSelectionMode(QTreeWidget.ExtendedSelection)

        self.tree_widget.setHeaderLabels(["name", "size", "date"])
        self.tree_widget.setStyleSheet("background-color: rgb(44,44,44);")
        
        header = self.tree_widget.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.resizeSection(1, header.sectionSize(2) // 2)
        
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(2, QHeaderView.Interactive)
        header.resizeSection(2, header.sectionSize(2) // 1.8)

        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.confirm_delete_selected_folder)  # Connect to confirmation function
        self.delete_button.setIcon(QIcon("/stdrepo/PFX/FXteamPath/icons/icons/trash.png"))
        self.refresh = QPushButton("Refresh")
        self.refresh.setIcon(QIcon("/stdrepo/PFX/FXteamPath/icons/icons/refresh.png"))
        self.refresh.clicked.connect(self.refreshTree)  # Connect to confirmation function
        layout.addWidget(self.tree_widget)
        self.tree_widget.setSortingEnabled(True)
        button_layout.addWidget(self.delete_button, 8)
        button_layout.addWidget(self.refresh, 2)
        layout.addLayout(button_layout)

dialog = HelloWindow()
dialog.show()

