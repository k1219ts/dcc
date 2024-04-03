# -*- coding:utf-8 -*-

import maya.cmds as cmds
import maya.mel as mm

def frameCounterUpdate():
    if cmds.headsUpDisplay("absframeCounter", q=1, ex=1) != 0:
        cmds.expression( s= "headsUpDisplay -r \"absframeCounter\";", n="frameCounterUpdate", ae=1, uc="all")
    else:
        cmds.delete("frameCounterUpdate")
        
def expressionRemover():
    if cmds.objExists('frameCounterUpdate') == 1:
        allExpr = cmds.ls("frameCounterUpdate", type = "expression")
        for curExpr in allExpr:
            cmds.delete(curExpr)
    
class HeadsUpDisplayMain:
    def __init__(self):

        ownerName = mm.eval('getenv "USERNAME"')
        nameTxtPath = "/home/" + ownerName + "/maya/name.txt"
        fileId = open(nameTxtPath, 'r')
        usrName = fileId.read()

        if cmds.window('HeadsUpDisplayWin', ex=1): cmds.deleteUI('HeadsUpDisplayWin')
        cmds.window('HeadsUpDisplayWin', title="HUD", sizeable=0)
        
        cmds.columnLayout('HeadsMainClm', adjustableColumn=True )
        cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[(1, 160)])
        
        cmds.frameLayout('mainFrame', l= "Enter The Details",
                                    fn="plainLabelFont",
                        			lv=1,
                        			mw=2,
                        			mh=2,
                        			bs="etchedIn")
                        			
        cmds.text(l="[ Artist Name ]")
                        			
        cmds.textField('Namefield', tx=usrName)
        
        #cmds.text(l="[ Scene Number ]")
                        			
        #cmds.textField('SceneNumField', tx=headFname)
        
        cmds.text("[ Current Status ]")
                        			
        cmds.optionMenu('statusMenu',#l= "Select", 
                            #backgroundColor = [0.706,0.194,0.194],
                            #backgroundColor = [1,1,1],
                            backgroundColor = [0.706,0.565,0.194],
                            #backgroundColor = [0.706,0.263,0.472],
                            w=150 )
        cmds.menuItem(l = "Final")
        
        cmds.setParent( 'HeadsMainClm' )	

        cmds.displayColor('headsUpDisplayLabels', 22)
        self.offAllHud()
        self.mg_removeHUD()
        expressionRemover()
        
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
                                event="idle", 
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
                                command = ("cmds.optionMenu('statusMenu', q=1, v=1)"))
                                
            cmds.headsUpDisplay('camName', l="Camera        ", allowOverlap = 1,
                                s=8,
                                b=1,
                                lfs = "large",
                                bs = "small",
                                preset = "cameraNames")


# ======================================================================================================================= #
    
    def mg_removeHUD(self, *args):
        expressionRemover()
        
    	if cmds.headsUpDisplay('versionName', exists=1):cmds.headsUpDisplay('versionName', rem=1)
    	
    	if cmds.headsUpDisplay('artistName', exists=1):cmds.headsUpDisplay('artistName', rem=1)
    	
    	if cmds.headsUpDisplay('ownerName', exists=1):cmds.headsUpDisplay('ownerName', rem=1)
    	
    	if cmds.headsUpDisplay('sceneName', exists=1):cmds.headsUpDisplay('sceneName', rem=1)
    	
    	if cmds.headsUpDisplay('dateName', exists=1):cmds.headsUpDisplay('dateName', rem=1)
    	
    	if cmds.headsUpDisplay('status', exists=1):cmds.headsUpDisplay('status', rem=1)
    	
    	#if cmds.headsUpDisplay('sceneNum', exists=1):cmds.headsUpDisplay('sceneNum', rem=1)
    	
    	if cmds.headsUpDisplay('frameCounter', exists=1):cmds.headsUpDisplay('frameCounter', rem=1)
    	
    	if cmds.headsUpDisplay('absframeCounter', exists=1):cmds.headsUpDisplay('absframeCounter', rem=1)

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
            
        if cmds.optionVar(q='focalLengthVisibility') == 1:
            mm.eval("ToggleFocalLength;")
            
        if cmds.optionVar(q='viewAxisVisibility') == 1:
            mm.eval("ToggleViewAxis;")
            
        if cmds.toggleAxis(q=1,o=1) == 1:
            mm.eval("ToggleOriginAxis;")
            
        if cmds.viewManip(q=1,v=1) == 1:
            mm.eval("ToggleViewCube;")
    	
# ======================================================================================================================= #
    	
#HeadsUpDisplayMain()
