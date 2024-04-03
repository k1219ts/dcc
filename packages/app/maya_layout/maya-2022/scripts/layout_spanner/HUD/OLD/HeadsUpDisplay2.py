# -*- coding:utf-8 -*-

import maya.cmds as cmds
import maya.mel as mm

createScriptNodeString = """
import HeadsUpDisplay as hud

if cmds.objExists('frameCounterUpdate') == 1:
    allExpr = cmds.ls("frameCounterUpdate*", type = "expression")
    for curExpr in allExpr:
        cmds.delete(curExpr)
        
hud.HeadsUpDisplayMain().offAllHud()
hud.HeadsUpDisplayMain().mg_removeHUD()
hud.expressionRemover()
"""

if not cmds.optionVar( ex="UserName" ):
    cmds.optionVar( sv = ( 'UserName', "NoName" ) )

if not cmds.optionVar( ex="ProgressSteps" ):
    cmds.optionVar( iv = ( 'ProgressSteps',  10 ) )

if not cmds.optionVar( ex="HUDStatusVar" ):
    cmds.optionVar( sv = ( 'HUDStatusVar',  "Blocking/Detail/Facial/Detail Facial/Final" ) )

STATUS = cmds.optionVar( q=  'HUDStatusVar').split("/")

def frameCounterUpdate():
    if cmds.headsUpDisplay("absframeCounter", q=1, ex=1) != 0:
        cmds.expression( s= "headsUpDisplay -r \"absframeCounter\";", n="frameCounterUpdate", ae=1, uc="all")
    else:
        cmds.delete("frameCounterUpdate")
        
def expressionRemover():
    if cmds.objExists('frameCounterUpdate*') == 1:
        allExpr = cmds.ls("frameCounterUpdate*", type = "expression")
        for curExpr in allExpr:
            cmds.delete(curExpr)

def CameraInfo():
    modelPanel = cmds.getPanel( withFocus = 1 )
    if cmds.getPanel( typeOf = modelPanel ) != "modelPanel":
        return ""
    cameraName = cmds.modelEditor( modelPanel, q = 1, camera = 1 )
    if cmds.nodeType( cameraName ) == "camera":
        buf = cmds.listRelatives( cameraName, parent = 1 )
        cameraName = buf[0]

    return cameraName

def FocalLengthInfo():
    camName = CameraInfo()
    if camName == "":
        return ""
    FL = cmds.getAttr( camName + ".focalLength" )
    return FL
    
class HeadsUpDisplayMain:
    def HeadUpDisPlayUI(self):
        
        if cmds.optionVar( q = "UserName" ) == "NoName":
            result = cmds.promptDialog(title='Enter New User Name',message='Enter Your Name',button=['OK', 'Cancel'],
                            text = '',
                            defaultButton='OK',
                            cancelButton='Cancel',
                            dismissString='Cancel')
            if result == 'OK':
                newUserName = str(cmds.promptDialog(query=True, text=True))
                cmds.optionVar( sv = ( 'UserName', newUserName ) )

        ProgressStep = cmds.optionVar( q="ProgressSteps" )


        if cmds.window('HeadsUpDisplayWin', ex=1): cmds.deleteUI('HeadsUpDisplayWin')
        if cmds.window('HUDstatusListWindow', ex=1): cmds.deleteUI('HUDstatusListWindow')
        cmds.window('HeadsUpDisplayWin', title="HUD", sizeable=0, menuBar = 1)
        cmds.menu( label='Edit', tearOff=True )
        cmds.menuItem( label='Edit Status', c = self.HUDstatusList )
        
        cmds.columnLayout('HeadsMainClm', adjustableColumn=True )
        cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[(1, 160)])
                
        cmds.text(l="Artist Name")
                                    
        cmds.textField( 'Namefield', tx= cmds.optionVar( q = 'UserName' ), cc = "cmds.optionVar( sv = ( 'UserName', cmds.textField( 'Namefield', q = 1, tx = 1 ) ) )" )
        
        cmds.separator( style = "none", h = 10)
                                    
        cmds.optionMenu('statusMenu', backgroundColor = [0.706,0.565,0.194], w=150 )
        cmds.menuItem(l = "------- Select Status -------")
        for tempStatus in STATUS:
            cmds.menuItem(l = tempStatus)
        
        cmds.separator( style = "in", h = 10)

        cmds.setParent( '..' )  
        
        cmds.rowColumnLayout(numberOfColumns=3, columnWidth=[(1, 60), (2, 70), (3, 30)])
        cmds.text(l = "Progress")
        cmds.optionMenu('progressMenu', backgroundColor = [0,0,0], w=70)
        cmds.menuItem(l = "Select")
        for tempProcess in range( 10, 101, ProgressStep ):
            cmds.menuItem( l = tempProcess )

        cmds.text(l = "%")
        
        cmds.setParent( '..' )
        
        cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[(1, 160)])
        cmds.separator( style = "in", h = 10)
        cmds.button('applyButton', l="Create Heads Up Display", backgroundColor = [.223, .559, .223],c = self.mg_CreateHUD)
        cmds.button('cancelButton', l="Remove Heads Up Display", c = self.mg_removeHUD)
        
        cmds.setParent( '..' )   

        cmds.window( 'HeadsUpDisplayWin', e = True, h = 180, w = 160)
        cmds.showWindow('HeadsUpDisplayWin')
        
# ======================================================================================================================= #
       
    def mg_CreateHUD(self, *args):
        if cmds.optionMenu('statusMenu', q=1, v=1) == "------- Select Status -------":
            cmds.confirmDialog( title='Confirm Status', message='Select Status', button=['Close'], cancelButton='Close', dismissString='Close' )
        else:
            if cmds.optionMenu('progressMenu', q=1, v=1) == "Select":
                cmds.confirmDialog( title='Confirm Progress', message='Select Progress', button=['Close'], cancelButton='Close', dismissString='Close' )
            else:
                cmds.displayColor('headsUpDisplayLabels', 22)
                cmds.displayColor('headsUpDisplayValues', 16)
                self.offAllHud()
                self.mg_removeHUD()
                expressionRemover()

                """
                if cmds.ls('HUIdelNode*') == []:
                    cmds.scriptNode( scriptType = 1, beforeScript = createScriptNodeString, name = 'HUIdelNode', sourceType = "python")
                """
                
                self.Fname = cmds.file(sceneName=True, shortName=True, q=True)
                
                if self.Fname == "":
                    cmds.error("Head Up Info : - load the inputs correctly !")
                else:
                    cmds.headsUpDisplay('artistName', l="Animator     ",allowOverlap = 1,
                                        b = 2,
                                        s = 5,
                                        lfs = "large",
                                        bs = "small",
                                        dataFontSize = "large",
                                        command=("cmds.textField('Namefield', q=1, tx=1)") )
                    
                    cmds.headsUpDisplay('sceneName', l="Scene Info   ",allowOverlap = 1,
                                        event="SceneOpened", 
                                        b = 3,
                                        s = 5,
                                        lfs = "large",
                                        bs = "small",
                                        dataFontSize = "large",
                                        command = ("cmds.file(q=1, namespace=1)"))
                        
                    cmds.headsUpDisplay('dateName', l = "Date | Time  ",allowOverlap = 1,
                                        #event="idle", 
                                        nodeChanges = "attributeChange",
                                        dataFontSize = "large",
                                        command = ('cmds.date(format="DD / MM / YYYY   |   hh:mm ")'),
                                        b = 4,
                                        s = 5,
                                        lfs = "large",
                                        bs = "small")
                                        
                    cmds.headsUpDisplay('absframeCounter', l= "Duration         ",allowOverlap = 1,
                                        b = 1,
                                        s = 6,
                                        lfs = "large",
                                        bs = "small",
                                        dataFontSize = "large",
                                        command = ("CurrentTime_ = cmds.currentTime(q=1);StartTime_ = cmds.playbackOptions(q=1, min=1);AbsTime_ = int(CurrentTime_ - StartTime_ + 1);AbsTime_"),
                                        event = "timeChanged") 
                                        
                    if cmds.objExists('frameCounterUpdate') != 1:
                        frameCounterUpdate()
                                        
                    cmds.headsUpDisplay('frameCounter', l= "Frame         ",allowOverlap = 1,
                                        b = 2,
                                        s = 6,
                                        lfs = "large",
                                        bs = "small",
                                        nodeChanges = "instanceChange",
                                        dataFontSize = "large",
                                        preset = "currentFrame")
                                              
                    cmds.headsUpDisplay('status', l="Status        ",allowOverlap = 1,
                                        b = 3,
                                        s = 6,
                                        lfs = "large",
                                        bs = "small",
                                        dataFontSize = "large",
                                        command = ("ComplitionProgress = cmds.optionMenu('statusMenu', q=1, v=1) + '   ' + cmds.optionMenu('progressMenu', q=1, v=1) + ' %';ComplitionProgress"))
                                        
                    cmds.headsUpDisplay('camName', l="Camera        ", allowOverlap = 1,
                                        s=8,
                                        b=1,
                                        lfs = "large",
                                        bs = "small",
                                        preset = "cameraNames")

                    cmds.headsUpDisplay('FocalLengthInfo', l="Focal Length        ", allowOverlap = 1,
                                        s=7,
                                        b=2,
                                        lfs = "small",
                                        bs = "small",
                                        dataFontSize = "small",
                                        command=( "hud.FocalLengthInfo()" ) )

# ======================================================================================================================= #
    
    def mg_removeHUD(self, *args):
        expressionRemover()
        
        if cmds.ls('HUIdelNode*') != []:
            cmds.delete( cmds.ls('HUIdelNode*') )
        
        if cmds.headsUpDisplay('versionName', exists=1):cmds.headsUpDisplay('versionName', rem=1)
        
        if cmds.headsUpDisplay('artistName', exists=1):cmds.headsUpDisplay('artistName', rem=1)
        
        if cmds.headsUpDisplay('ownerName', exists=1):cmds.headsUpDisplay('ownerName', rem=1)
        
        if cmds.headsUpDisplay('sceneName', exists=1):cmds.headsUpDisplay('sceneName', rem=1)
        
        if cmds.headsUpDisplay('dateName', exists=1):cmds.headsUpDisplay('dateName', rem=1)
        
        if cmds.headsUpDisplay('status', exists=1):cmds.headsUpDisplay('status', rem=1)
        
        if cmds.headsUpDisplay('frameCounter', exists=1):cmds.headsUpDisplay('frameCounter', rem=1)
        
        if cmds.headsUpDisplay('absframeCounter', exists=1):cmds.headsUpDisplay('absframeCounter', rem=1)
        
        if cmds.headsUpDisplay('camName', exists=1):cmds.headsUpDisplay('camName', rem=1)

        if cmds.headsUpDisplay('FocalLengthInfo', exists=1):cmds.headsUpDisplay('FocalLengthInfo', rem=1)
        

# ======================================================================================================================= #
    
    def offAllHud(self, *args):
        """
        buf_ = cmds.headsUpDisplay(lh=1)
        if buf_ != None:
            for hudList in buf_: cmds.headsUpDisplay(hudList, rem=1)
        """    
        if cmds.optionVar(q='selectDetailsVisibility') == 1:
            mm.eval("ToggleSelectDetails;")
            
        if cmds.optionVar(q='objectDetailsVisibility') == 1:
            mm.eval("ToggleObjectDetails;")
            
        if cmds.optionVar(q='polyCountVisibility') == 1:
            mm.eval("TogglePolyCount;")
            
        if cmds.optionVar(q='subdDetailsVisibility') == 1:
            mm.eval("ToggleSubdDetails;")
            
        if cmds.optionVar(q='animationDetailsVisibility') == 1:
            mm.eval("ToggleAnimationDetails;")
            
        if cmds.optionVar(q='fbikDetailsVisibility') == 1:
            mm.eval("ToggleFbikDetails;")
            
        if cmds.optionVar(q='frameRateVisibility') == 1:
            mm.eval("ToggleFrameRate;")
            
        if cmds.optionVar(q='currentFrameVisibility') == 1:
            mm.eval("ToggleCurrentFrame;")
            
        if cmds.optionVar(q='sceneTimecodeVisibility') == 1:
            mm.eval("ToggleSceneTimecode;")
            
        if cmds.optionVar(q='currentContainerVisibility') == 1:
            mm.eval("ToggleCurrentContainerHud;")
            
        if cmds.optionVar(q='cameraNamesVisibility') == 1:
            mm.eval("ToggleCameraNames;")
            
        # edit Moon ===========================================
        if cmds.optionVar(q='focalLengthVisibility') == 1:
            mm.eval("ToggleFocalLength;")
        # =====================================================    

        if cmds.optionVar(q='viewAxisVisibility') == 1:
            mm.eval("ToggleViewAxis;")
            
        if cmds.toggleAxis(q=1,o=1) == 1:
            mm.eval("ToggleOriginAxis;")
            
        if cmds.viewManip(q=1,v=1) == 1:
            mm.eval("ToggleViewCube;")


    def HUDstatusList(self,  *args):

        if cmds.window('HUDstatusListWindow', ex=1):cmds.deleteUI('HUDstatusListWindow')
        cmds.window('HUDstatusListWindow', title="Edit Items", sizeable=1)
        cmds.columnLayout(adjustableColumn=True )

        cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[(1, 160)])

        cmds.textScrollList('HUDStatusList', numberOfRows = 8, allowMultiSelection=True,
                    append = STATUS,
                    showIndexedItem=4, height = 100, font = "fixedWidthFont" )
                    
                    
        cmds.setParent('..')

        cmds.rowColumnLayout(numberOfColumns = 2, columnWidth=[(1, 80), (2, 80)])          

        cmds.button('AddItemBtn', l = 'ADD', c = self.AddStatusItem )
        cmds.button('DelItemBtn', l = 'DEL', c = self.DelStatusItem )
        
        cmds.setParent('..')
        
        cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[(1, 160)])
            
        cmds.button(l = "Close", c = "cmds.deleteUI('HUDstatusListWindow');import HeadsUpDisplay2 as hud;reload(hud);hud.HeadsUpDisplayMain().HeadUpDisPlayUI()")

        cmds.window('HUDstatusListWindow', e = 1, wh = (165, 150))
                    
        cmds.showWindow('HUDstatusListWindow')


    def AddStatusItem(self, *args):
        HUDStatusItemList = cmds.textScrollList('HUDStatusList', q = 1, allItems = 1)
        SelectedItem = cmds.textScrollList('HUDStatusList', q = 1, selectItem = 1)
        
        SelectedItemIndex = HUDStatusItemList.index( SelectedItem[0] )
        
        result = cmds.promptDialog(
            title='Add Status',
            message='Enter New Status:',
            button=['OK', 'Cancel'],
            defaultButton='OK',
            cancelButton='Cancel',
            dismissString='Cancel')
            
        if result == 'OK':
            NewItem = cmds.promptDialog(query=True, text=True)
            cmds.textScrollList('HUDStatusList', e = 1, appendPosition = (SelectedItemIndex + 2, NewItem) )

            NewList = cmds.textScrollList('HUDStatusList', q = 1, allItems = 1)
            cmds.optionVar( sv = ( 'HUDStatusVar',   "/".join( NewList ) ) )
            
    def DelStatusItem(self, *args):
        SelectedItem = cmds.textScrollList('HUDStatusList', q = 1, selectItem = 1)
        cmds.textScrollList('HUDStatusList', e = 1, removeItem = SelectedItem[0] )
        NewList = cmds.textScrollList('HUDStatusList', q = 1, allItems = 1)
        cmds.optionVar( sv = ( 'HUDStatusVar',   "/".join( NewList ) ) )
    
    
    


        
# ======================================================================================================================= #
        
#HeadsUpDisplayMain().HeadUpDisPlayUI()
