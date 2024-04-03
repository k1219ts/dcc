# -*- coding: utf-8 -*-

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import hou, time, os, shutil, sys
import time, dxConfig, requests, subprocess, getpass, shutil

TACTIC_IP = dxConfig.getConf("TACTIC_IP") #Tactic ip retrieved from dxConfig class getConf func
API_KEY = "c70181f2b648fdc2102714e8b5cb344d" #API_KEY. hard coded
shotServer = "http://%s/dexter/search/shot.php" %(TACTIC_IP) #shot Server 

class Butler(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent, Qt.WindowStaysOnTopHint)
        self.initUI()
        
        
        
    def freeCacheSpace(self):
        v = hou.applicationVersion
        if str(v()[0])=='18':
            return "Cache size not supported in Houdini 18.5"
        elif str(v()[0])=='20':
            os.statvfs('/fx_cache')
            total, used, free = shutil.disk_usage('/fx_cache/')
            freespace = "FX Cache Free Space : "+"{:.2f}".format((free / total) * 100)+"%"
            return freespace
        
        
    def _hipEmpty(self):
        childCnt = hou.node("/obj").allSubChildren()
        
        if len(childCnt) == 0:
            print("File Check : Child Count is 0")
            return True
        if len(childCnt) !=0:
            return False
        
            
    def openFolder(self,showcode,showname,seq,shot,which):
    
        if which == 'hip' :
            path='/show/'+str(showname)+'/works/PFX/shot/' + str(seq) + '/' + str(shot) + '/dev/scenes/'
            subprocess.Popen(["xdg-open",path])
        elif which == 'fxcache' :
            path='/fx_cache/'+str(showname)+'/' + str(seq) + '/' + str(shot) + '/dev/'
            subprocess.Popen(["xdg-open",path])
            
            
    def speedCheck(self,showcode,showname,seq,shot):
    
        info = requests.get(shotServer,params={'api_key':API_KEY,'project_code':showcode,'code':shot}).json()
        stF= info[0]['frame_in']
        endF = info[0]['frame_out']
        tasks = info[0]['tasks']

        print("---DEBUG---")
        print("Current user :" + str(getpass.getuser()))
        print("Start Frame :" + str(stF))
        print("End Frame :" + str(endF))

        if self._hipEmpty() == True:
            print("We have a liftoff")
            self.configureShot(showcode,showname,seq,shot)
        elif self._hipEmpty() == False:
            print("Empty your scene first")
        
            
    def configureShot(self,showcode,showname,seq,shot):
        stF=self.prerollBox.text()
        info = requests.get(shotServer,params={'api_key':API_KEY,'project_code':showcode,'code':shot}).json()
        prependstF=int(stF)
        stF= info[0]['frame_in']
        endF = info[0]['frame_out']

        if self.checkBox.isChecked(): #check if user requires pre-roll
            stF=stF-prependstF
            hou.playbar.setFrameRange(stF,endF)
            hou.setFrame(1001)
        else :
            hou.playbar.setFrameRange(stF,endF)
            hou.setFrame(stF)
        
        self.misc_Nodes(showcode,showname,seq,shot,stF,endF)       

        
    def misc_Nodes(self,showcode,showname,seq,shot,stF,endF):

        sceneSetting = hou.node("/obj").createNode("DX_Scene_Setting")
        parm=sceneSetting.parm("f1")
        parm.deleteAllKeyframes()
        parm=sceneSetting.parm("f2")
        parm.deleteAllKeyframes()
        sceneSetting.setParms({"trange":1, "f1":1001, "f2":endF})
        sceneSetting.parm("Set").pressButton()
        sceneSetting.setParms({"scale":float(self.scleBox.text())})
        sceneSetting.setComment(shot)
        sceneSetting.setGenericFlag(hou.nodeFlag.DisplayComment,True)

        
        impUSDIN="USD_IN"
    
        OBJ=hou.node("/obj")
        #Create Import
        geo = OBJ.createNode("geo")
        geo.setName("import")
        geo.setInput(0, hou.node("/obj/DX_Scene_Setting1"))
        geo.moveToGoodPosition()
        geo.move(hou.Vector2((2,-0.5)))

        
        #Create lopcam
        geo = OBJ.createNode("lopimportcam")
        geo.setInput(0, hou.node("/obj/DX_Scene_Setting1"))
        geo.setName("maincam")
        geo.moveToGoodPosition()
        geo.move(hou.Vector2((-2,-0.5)))
        
        geo.setParms({"loppath":"/obj/import/USD_IN/LOPNET_for_Import/to_import"})
        geo.setParms({"primpath":"/World/Cam/main_cam"})
        #geo.parm("resy").deleteAllKeyframes()
        geo.setParms({"resx":2048})
        geo.parm("shutter").deleteAllKeyframes()
        geo.setParms({"vm_background":'/show/'+str(showname)+'/_2d'+'/shot/'+str(seq)+'/'+str(shot)})

        #Create dxcamfrustum
        geo = OBJ.createNode("DX_camFrustum")
        geo.setInput(0, hou.node("/obj/maincam"))
        geo.moveToGoodPosition()
        geo.move(hou.Vector2((0,-0.5)))
        geo.setColor(hou.Color(1,0,0))
        
        #Create USD_In
        imP=hou.node("/obj/import")
        geo = imP.createNode("usdimport")
        geo.setName(impUSDIN)
        geo.moveToGoodPosition()
        Usdpath='/show/'+str(showname)+'/_3d'+'/shot/'+str(seq)+'/'+str(shot)+'/'+str(shot)+'.usd'
        
        geo.setParms({"primpattern":""})
        geo.setParms({"filepath1":Usdpath})
        geo = imP.createNode("null")
        geo.setName("USD_OUT")
        geo.setInput(0, hou.node("/obj/import/USD_IN"))
        geo.moveToGoodPosition()
        geo.move(hou.Vector2((0,-0.5)))
        
        #Create WORK
        OBJ=hou.node("/obj")
        work = OBJ.createNode("subnet")
        work.setName("WORK")
        work.move(hou.Vector2((0.5,-2.3)))   
        work.setColor(hou.Color(1,0,0))
        work.setComment(shot)
        work.setGenericFlag(hou.nodeFlag.DisplayComment,True)
        
        for i in hou.node("/obj/WORK").allItems():
            i.setPosition(hou.Vector2(0,0))
            i.setColor(hou.Color(0,0,0))
            
        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("ropnet")
        work.setName("render")
        
        rnPos = work.position()
        work.moveToGoodPosition()
        work.move(hou.Vector2(0,-3))
        rnPos = work.position()
        OBJ=hou.node("/obj/WORK/render")
        
        v = hou.applicationVersion
        
        if str(v()[0])=='18':
            work = OBJ.createNode("DXK_mantra")
        elif str(v()[0])=='20':
            work = OBJ.createNode("yj::DXC_Mantra::1.1")
            
        try:  
            work.setName("REN")
            work.setParms({"camera":"/obj/maincam","trange":1,"allowmotionblur":1,"vobject":""})
        except:
            pass
        
        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("ropnet")
        work.setName("data")
        work.move(rnPos)
        work.move(hou.Vector2(0,-1))
        
        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("ropnet")
        work.setName("sim")
        work.move(rnPos)
        work.move(hou.Vector2(0,-2))
        
        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("geo")
        work.setName("FX")
        work.move(rnPos)
        work.move(hou.Vector2(-5,0))
        work.setColor(hou.Color(1,0,0))
        
        fxPos=work.position()

        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("geo")
        work.setName("REN_")
        work.move(fxPos)
        work.move(hou.Vector2(0,-7))
        work.setColor(hou.Color(0.451,0.369,0.796))
        work.setParms({"geo_velocityblur":1})
        
        OBJ=hou.node("/obj/WORK")
        work = OBJ.createNode("matnet")
        work.setName("material")
        work.move(rnPos)
        work.move(hou.Vector2(0,2))
        
        OBJ=hou.node("/obj/WORK/REN_")
        work = OBJ.createNode("object_merge")
        work.setName("OBJ_IN")
        
        work.setColor(hou.Color(0.451,0.369,0.796))

        OBJ=hou.node("/obj/WORK/FX")
        work = OBJ.createNode("object_merge")
        work.setName("OBJ_IN")
        work.setParms({"objpath1":"/obj/import/USD_OUT"})
        work.setColor(hou.Color(0.451,0.369,0.796))
        inPos=work.position()
        work = OBJ.createNode("null")
        work.setName("OUT_")
        work.setColor(hou.Color(0.451,0.369,0.796))
        work.move(inPos)
        work.move(hou.Vector2(0,-12))
        
    def dbSearch(self):
        try:
            hipname_ = hou.hipFile.name()
            show = hipname_.split("/")[2]
            seq = hipname_.split("/")[6]
            shot = hipname_.split("/")[7]    
            requestParam = dict() # eqaul is requestParm = {}
            requestParam['api_key'] = API_KEY
            requestParam['name'] = show
            responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC_IP), params=requestParam)
            showname = responseData.json()[0]['name'] #working name
            show_descriptive_name = responseData.json()[0]['description'] #Veloz
            showcode = responseData.json()[0]['code']
    
            return showcode,showname,seq,shot,show_descriptive_name
        except IndexError:
            self.yjDialog()
        
    def yjDialog(self):
        message = "Pardon, are you sure this is a shot file?"
        confirm_dialog = QMessageBox()
        confirm_dialog.setIcon(QMessageBox.Warning)
        confirm_dialog.setWindowTitle("Butler Warning")
        confirm_dialog.setText(message)
        confirm_dialog.setStandardButtons(QMessageBox.Ok)
        confirm_dialog.setButtonText(QMessageBox.Ok,"      Let me check      ")
        curs=QCursor.pos()
        confirm_dialog.move(curs.x(),curs.y())
        result = confirm_dialog.exec_()
        sys.exit()
        
    def togglePreroll(self):
        if self.checkBox.isChecked():
            self.prerollBox.setEnabled(True)
        else:
            self.prerollBox.setEnabled(False)
    
    def initUI(self):

        showcode,showname,seq,shot,show_descriptive_name = self.dbSearch()
        
        self.setWindowTitle('FX Bulter, At your service v1.0')
        self.setGeometry(300, 300, 500, 400) # x, y, width, height
        self.label = QLabel(self)
        logo = QPixmap("/stdrepo/PFX/Artist/yongjun.cho/script_asset/butler_logo_inverted.png")
        logo = logo.scaledToWidth(500)
        layout_logo = QHBoxLayout()
        layout = QVBoxLayout(self)
        layout_first=QHBoxLayout()
        layout_second=QHBoxLayout()
        layout_third=QHBoxLayout()
        layout_fourth=QHBoxLayout()
        
        self.label.setPixmap(logo)
        self.label.setAlignment(Qt.AlignCenter)
        self.label_show = QLabel("   Project : "+show_descriptive_name, self)
        self.label_seq = QLabel("     Sequence No. : "+seq, self)
        self.label_shot = QLabel("   Shot No. : "+shot, self)
        
        font = QFont()
        font.setBold(True)
        
        self.label_show.setFont(font)
        self.label_seq.setFont(font)
        self.label_shot.setFont(font)

        self.set_button = QPushButton("  Prepare my shot  ")
        
        self.hip_button = QPushButton("Open hip file folder")
        self.fxcache_button = QPushButton("Open Cache folder")

        self.prerollBox = QLineEdit()
        self.scleBox = QLineEdit()
        
        self.prerollLabel = QLabel(self)
        self.prerollLabel.setText("Preroll Amount : ")
        
        self.freespaceLabel = QLabel(self)
        self.freespaceLabel.setText(self.freeCacheSpace())
        
        self.scleLabel = QLabel(self)
        self.scleLabel.setText("Scene Scale : ")
        self.nullLabel = QLabel(self)
        self.nullLabel.setText("")
        
        self.scleBox.setText("0.1")
        
        self.prerollBox.setText("20")
        self.checkBox = QCheckBox('Pre roll?',self)
        self.prerollBox.setEnabled(False)
        self.checkBox.setChecked(0)

        self.checkBox.clicked.connect(lambda: self.togglePreroll())
        self.set_button.clicked.connect(lambda: self.speedCheck(showcode,showname,seq,shot))  # Connect to confirmation function
        self.hip_button.clicked.connect(lambda: self.openFolder(showcode,showname,seq,shot,which='hip'))  # Connect to confirmation function
        self.fxcache_button.clicked.connect(lambda: self.openFolder(showcode,showname,seq,shot,which='fxcache'))  # Connect to confirmation function
        
        hline3=QFrame()
        hline3.setFrameShape(QFrame.HLine)
        hline3.setFrameShadow(QFrame.Sunken)
        hline3.setStyleSheet("background-color: grey;")
        
        layout_second.addWidget(self.label_show, 3)
        layout_second.addWidget(self.label_seq, 3)
        layout_second.addWidget(self.label_shot, 3)

        
        layout_logo.addWidget(self.label)
        layout_third.addWidget(self.checkBox,1)
        layout_third.addWidget(self.prerollLabel,1)
        layout_third.addWidget(self.prerollBox,1)
        
        layout_third.addWidget(self.scleLabel,1)
        layout_third.addWidget(self.scleBox,1)
        layout_first.addWidget(self.set_button, 5) #add set button
        
        if self._hipEmpty() == True :
            self.hip_button.hide()
            self.fxcache_button.hide()
            self.freespaceLabel.hide()
            pass
        elif self._hipEmpty() == False :
            self.set_button.setDisabled(True)
            layout_fourth.addWidget(self.hip_button, 5) #add set button
            layout_fourth.addWidget(self.fxcache_button, 5) #add set button
            self.freespaceLabel.setFont(font)
            layout_fourth.addWidget(self.freespaceLabel, 3)
            
        layout.setSpacing(30)
        layout.addLayout(layout_logo)
        layout.addLayout(layout_second)
        layout.addLayout(layout_third)
        layout.addWidget(hline3)
        layout.addLayout(layout_first)
        layout.addLayout(layout_fourth)

        
dialog = Butler()
dialog.show()

