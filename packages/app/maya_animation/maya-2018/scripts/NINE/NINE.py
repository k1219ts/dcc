#encoding:utf-8
#==============================
# line_rig_tool
#==============================

import maya.cmds as cmds
import maya.mel as mel




class class_NINE:
    
    ##########################################################################################################################################################################################################
    ## window UI
    ##########################################################################################################################################################################################################
    
    def __init__(self):
        
        self.title = 'NINE_rig_tool'
        
        start_frame = int( cmds.playbackOptions( q=1, min=1 ) -1 )
        end_frame = int( cmds.playbackOptions( q=1, max=1 ) + 1 )

        if cmds.window ( self.title, q=1, ex=1 ):
            cmds.deleteUI ( self.title )
        
        cmds.window ( self.title, t='NINE ver 1.1', s=0 )
        cmds.columnLayout( 'maincolumn', co=('both', 8) )
            
        cmds.separator( st="in" )
        
        
        ### Rig Tap ###
        cmds.frameLayout ( 'controler_rig', label='Controler Rigging', bgc=[ 0.2, 0.3, 0.42 ], cll=1, cl=1, w=460, mh=10, p='maincolumn' )

        cmds.rowColumnLayout ( nc=6, cw=[(1, 130), (2, 20), (3, 120), (4, 20), (5, 120), (6, 20)] ) 
        cmds.text ( 'Controler Type', bgc=[0.30, 0.40, 0.5] )
        cmds.text ( l='' )
        cmds.optionMenu( 'controler_type', label='', cc=self.controler_type_win_switch )
        cmds.menuItem( label = ' FK' )
        cmds.menuItem( label = ' IK' )
        cmds.menuItem( label = ' FK - IK' )
        cmds.menuItem( label = ' Path-IK - IK' )
        cmds.menuItem( label = ' Add B-FK' )
        cmds.text ( l='' )
        self.setting_num_tx = cmds.text ( 'setting_num_tx', l='   Option Disabled', en=0 )
        self.setting_num_tf = cmds.textField ( 'setting_num_tf', tx='4', en=0 )
        cmds.setParent('..')

        cmds.rowColumnLayout ( nc=6, cw=[(1, 130), (2, 20), (3, 120), (4, 25), (5, 140), (6, 20)] ) 
        cmds.text ( 'Rigging Type', bgc=[0.30, 0.40, 0.5] )
        cmds.text ( l='' )
        cmds.optionMenu( 'rigging_type', label='', cc=self.rigging_type_win_switch )
        cmds.menuItem( label = ' Controler' )
        cmds.menuItem( label = ' Curve Bind' )
        cmds.menuItem( label = ' Spline-IK' )
        cmds.menuItem( label = ' Hair Dynamic' )
        cmds.text ( l='' ) 
        self.upAxis_rig_option = cmds.checkBox( 'upAxis_rig_Option', l='Add Up-Axis Control Rig', en=0, v=0 )
        cmds.setParent('..')

        cmds.rowColumnLayout ( nc=3, cw=[(1, 130), (2, 160), (3, 150)] )
        cmds.text ( 'Tool Operate' )
        cmds.text ( '  Selct Joint or Curve  ' )
        cmds.button ( l='Apply', c=self.rig_OP )
        cmds.setParent('..')
        

        ### Instance Rig Tap ###
        cmds.frameLayout ( 'add_hair_dynamic', label='Instance Rigging', bgc=[ 0.3, 0.3, 0.4 ], cll=1, cl=0, w=460, mh=10, p='maincolumn' )        
        
        cmds.rowColumnLayout ( nc=6, cw=[(1, 130), (2, 20), (3, 120), (4, 20), (5, 120), (6, 20)] )
        cmds.text ( 'Instance Rig Type', bgc=[0.4, 0.4, 0.5] )
        cmds.text ( l='' )
        instance_rigging_type_PM = cmds.optionMenu( 'instance_rigging_type', label='', cc=self.IT_rigging_type_win_switch )
        cmds.menuItem( label = ' Path-IK Rig' )
        cmds.menuItem( label = ' FK Rig' )
        cmds.menuItem( label = ' FK Rig - Adance' )
        cmds.menuItem( label = ' HairSIM Rig - Simple' )
        cmds.menuItem( label = ' HairSIM Rig - Tip Con' )  
        cmds.menuItem( label = ' HairSIM Rig - Advance' )        
        cmds.text ( l='' )
        self.IT_setting_num_tx = cmds.text ( 'IT_setting_num_tx', l='  Option Disabled', en=0 )
        self.IT_setting_num_tf = cmds.textField ( 'IT_setting_num_tf', tx='4', en=0 )
        cmds.setParent('..')
        
        
        cmds.rowColumnLayout ( nc=8, cw=[( 1, 130), (2, 20), (3, 50), (4, 20), (5, 50), (6, 20), (7, 75), (8, 75)] )
        cmds.text ( l='Time Range' )
        cmds.text ( l='' )
        self.tx_startFrame = cmds.textField ( 'tx_startFrame', tx=start_frame, en=0 )
        cmds.text ( l='~' )
        self.tx_endFrame = cmds.textField ( 'tx_endFrame', tx=end_frame, en=0 )
        cmds.text ( l='' )
        cmds.button ( l='Apply', c=self.intance_rig_OP, ann=' 컨트롤러 생성\n  - 리깅을 따라갈 컨트롤러와 리깅을 생성할 컨트롤러를 순서대로 선택하고 실행하세요  \n  - Path-IK 리깅과 헤어시뮬레이션 리깅은 5개 이상의 컨트롤러를 선택하셔야 합니다  \n  ( Time Range를 기준으로 Bake가 됩니다 )  ' )
        cmds.button ( l='Bake', c=self.bake_controler, ann=' Bake\n - 맨 끝에 위치한 인스턴스 컨트롤러를 선택하고 실행하세요  \n  ( Time Range를 기준으로 Bake가 됩니다 )  ' )
        cmds.setParent('..')
        
        cmds.showWindow( self.title )
        
        cmds.optionMenu ( instance_rigging_type_PM, e=1, v=' HairSIM Rig - Advance' )
        

    

    ### rigging option UI On/Off ###    
    def controler_type_win_switch(self, *args):
        controler_type_value = cmds.optionMenu ( 'controler_type', q=1, v=1 )
        
        if controler_type_value ==' Path-IK - IK' or controler_type_value ==' Add B-FK':
            
            if controler_type_value ==' Path-IK - IK':
                cmds.text ( self.setting_num_tx, e=1, l=' Controler Number' )
            
            else:
                cmds.text ( self.setting_num_tx, e=1, l='Controler Interval' )   
            
            cmds.text ( self.setting_num_tx, e=1, en=1 )
            cmds.textField ( self.setting_num_tf, e=1, en=1 )

        else:
            cmds.text ( self.setting_num_tx, e=1, l='   Option Disabled' )
            cmds.text ( self.setting_num_tx, e=1, en=0 )
            cmds.textField ( self.setting_num_tf, e=1, en=0 )

            

    def rigging_type_win_switch(self, *args):
        rigging_type_win_value = cmds.optionMenu ( 'rigging_type', q=1, v=1 )

        if rigging_type_win_value ==' Spline-IK' or rigging_type_win_value ==' Hair Dynamic':
            cmds.checkBox ( self.upAxis_rig_option, e=1, en=1, v=0 )
    
        else:
            cmds.checkBox ( self.upAxis_rig_option, e=1, en=0, v=0 )
    


    def IT_rigging_type_win_switch(self, *args):
        IT_rigging_type_win_value = cmds.optionMenu ( 'instance_rigging_type', q=1, v=1 )
        
        if IT_rigging_type_win_value ==' Path-IK Rig' or IT_rigging_type_win_value ==' FK Rig - Adance':   
            
            if IT_rigging_type_win_value ==' Path-IK Rig':
                cmds.text ( self.IT_setting_num_tx, e=1, l=' Controler Number' )
            
            else  :
                cmds.text ( self.IT_setting_num_tx, e=1, l='Controler Interval' )
            
            cmds.text ( self.IT_setting_num_tx, e=1, en=1 )
            cmds.textField ( self.IT_setting_num_tf, e=1, en=1 )

        else:
            cmds.text ( self.IT_setting_num_tx, e=1, l='  Option Disabled' )
            cmds.text ( self.IT_setting_num_tx, e=1, en=0 )
            cmds.textField ( self.IT_setting_num_tf, e=1, en=0 )		




    ### 컨트롤러 크기 조절 ###
    def controler_scale(self, CRV_list, scale_value ):

        for x in CRV_list:
            CRV_shape_name = cmds.listRelatives (x, s=1)[0]
            CRV_span_num = cmds. getAttr ( CRV_shape_name+'.spans' )    
            cmds.select ( x+'.cv[0:%s]' %(CRV_span_num)) 
            cmds.scale ( scale_value, scale_value, scale_value, r=1 )
            
        cmds.select ( cl=1 )


   

    ### 조인트 생성 모듈 ###  return joint_list
    def joint_generate( self, kind ):
        
        cmds.select ( cl=1 )
        root_joint = cmds.joint ( p=self.point_list[0], n=kind+'_%s1_JNT' %( self.splitName ) )

        for x in range(self.cycle_num-1):
            inexNum =x+1
            cJoint = cmds.joint ( p=self.point_list[inexNum], n=kind+'_%s%s_JNT' %( self.splitName, (x+2) ) )
            pJoint = cmds.listRelatives ( p=1 )
            cmds.joint ( pJoint, e=1, zso=1, oj='xyz', sao='yup')
            cmds.select ( cJoint )
            

        cmds.select ( root_joint, hi=1 )
        joint_list = cmds.ls ( sl=1 )
        
        end_joint = joint_list[-1]
        cmds.setAttr ( end_joint+'.jointOrientX', 0)
        cmds.setAttr ( end_joint+'.jointOrientY', 0)
        cmds.setAttr ( end_joint+'.jointOrientZ', 0)
        
        return joint_list




    ### 커브나 조인트를 생성 ###  self.sel_curve, self.sel_curveShape, self.splitName, self.point_list, self.joint_list, self.cycle_num, self.LineRig_GRP 
    def jointCurve(self):

        sel_obj = cmds.ls ( sl=1 )[0]
        node_type = cmds.nodeType ( sel_obj )
        

        if node_type=='transform':
            self.sel_curve = cmds.ls ( sl=1 )[0]
            self.sel_curveShape = cmds.listRelatives ( self.sel_curve, s=1 )[0]
            self.splitName = self.sel_curve.split('_CRV')[0]
            cvpNum = self.cycle_num = ( cmds.getAttr ( '%s.degree' %(self.sel_curve) ) ) + ( cmds.getAttr ( '%s.spans' %(self.sel_curve) ) )
            self.point_list = []

            for x in range(cvpNum):
                each_point = cmds.pointPosition ( self.sel_curve+'.cv[%s]' %(x), w=1 )
                self.point_list.append(each_point)

            self.joint_list = self.joint_generate( 'SIM' )

        elif node_type=='joint':
            root_joint = cmds.ls ( sl=1 )[0]
            self.splitName = root_joint.split('_JNT')[0]
            cmds.select ( hi=1 )
            self.joint_list = cmds.ls ( sl=1 )
            self.cycle_num = len( self.joint_list )
            
            self.point_list = []

            for x in range( self.cycle_num ):
                self.point_list.append( cmds.xform ( self.joint_list[x], q=1, ws=1, t=1 ) )

            self.sel_curve = cmds.curve ( p=self.point_list, n='base_curve' )
            self.sel_curveShape = cmds.listRelatives ( self.sel_curve, s=1 )[0]
            

        self.LineRig_GRP = cmds.group ( n=self.splitName+'_LineRig_node_GRP', em=1 )
        cmds.parent ( self.joint_list[0], self.sel_curve, self.LineRig_GRP )





    ##########################################################################################################################################################################################################
    ## Generate Controler - self.sphere_CON_list, self.sphere_NUL_list, self.circle_CON_list, self.circle_NUL_list
    ##########################################################################################################################################################################################################
    
    def greate_controler( self, CON_shape, CON_color, parent_list, name ):

        if CON_shape == 'circle':
            self.circle_NUL_list = []
            self.circle_CON_list = []

            for x in range(len(parent_list)):
                circle_CON = ( mel.eval ( 'circle -c 0 0 0 -nr 1 0 0 -sw 360 -r 2 -d 3 -ut 0 -tol 0.01 -s 8 -ch 0 -n "%s%s_CON" ' %( name, (x+1) ) )[0] )
                circle_CONShape = cmds.listRelatives ( s=1 )[0]
                cmds.setAttr ( circle_CONShape+'.overrideEnabled', 1 )
                cmds.setAttr ( circle_CONShape+'.overrideColor', CON_color )
                self.circle_CON_list.append ( circle_CON )
    
                cmds.group ( n='%s%s_extra_NUL' % ( name, (x+1) ) )
                each_circle_NUL = cmds.group ( n='%s%s_NUL' % ( name, (x+1) ) )
                self.circle_NUL_list.append ( each_circle_NUL )    
                cmds.delete ( cmds.parentConstraint ( parent_list[x], each_circle_NUL ) )
                
            instance_circle_NUL_list = [] + self.circle_NUL_list
            instance_circle_CON_list = [] + self.circle_CON_list

            for x in range( len(parent_list)-1 ):
                cmds.parent ( instance_circle_NUL_list[-1], instance_circle_CON_list[-2] )
                del instance_circle_NUL_list[-1], instance_circle_CON_list[-1]
                
        elif CON_shape == 'sphere':
            self.sphere_NUL_list = []
            self.sphere_CON_list = []
            
            for x in range(len(parent_list)):          
                sphere_CON =  mel.eval ( '''curve -d 1 -p 0 0 1 -p 0 0.5 0.866025 -p 0 0.866025 0.5 -p 0 1 0 -p 0 0.866025 -0.5 -p 0 0.5 -0.866025 -p 0 0 -1 -p 0 -0.5 -0.866025
                                                -p 0 -0.866025 -0.5 -p 0 -1 0 -p 0 -0.866025 0.5 -p 0 -0.5 0.866025 -p 0 0 1 -p 0.707107 0 0.707107 -p 1 0 0 -p 0.707107 0 -0.707107
                                                -p 0 0 -1 -p -0.707107 0 -0.707107 -p -1 0 0 -p -0.866025 0.5 0 -p -0.5 0.866025 0 -p 0 1 0 -p 0.5 0.866025 0 -p 0.866025 0.5 0 -p 1 0 0
                                                -p 0.866025 -0.5 0 -p 0.5 -0.866025 0 -p 0 -1 0 -p -0.5 -0.866025 0 -p -0.866025 -0.5 0 -p -1 0 0 -p -0.707107 0 0.707107 -p 0 0 1
                                                -k 0 -k 1 -k 2 -k 3 -k 4 -k 5 -k 6 -k 7 -k 8 -k 9 -k 10 -k 11 -k 12 -k 13 -k 14 -k 15 -k 16 -k 17 -k 18 -k 19 -k 20 -k 21 -k 22 -k 23 -k 24
                                                -k 25 -k 26 -k 27 -k 28 -k 29 -k 30 -k 31 -k 32 -n "%s%s_move_CON" ''' %( name, (x+1) ) )
    
                sphere_CONShape = cmds.listRelatives ( sphere_CON, s=1 )[0]
                cmds.setAttr ( sphere_CONShape+'.overrideEnabled', 1 )
                cmds.setAttr ( sphere_CONShape+'.overrideColor', CON_color )
    
                self.sphere_CON_list.append ( sphere_CON )
                cmds.group ( n='%s%s_move_extra_NUL' %( name, (x+1) ) )
                each_sphere_NUL = cmds.group ( n='%s%s_move_NUL' % ( name, (x+1) ) )
                self.sphere_NUL_list.append ( each_sphere_NUL )           

                cmds.delete ( cmds.parentConstraint ( parent_list[x], each_sphere_NUL ) )
                
            
        elif CON_shape == 'hexagon':
            self.hexagon_NUL_list = []
            self.hexagon_CON_list = [] 
            
            for x in range(len(parent_list)):  
                hexagon_CON = mel.eval ( '''curve -d 1 -p -0.257187 0 0.445461 -p 0.257187 0 0.445461 -p 0.514375 0 2.51218e-07 -p 0.257187 0 -0.445461 -p -0.257187 0 -0.445461 -p -0.514375 0 1.69509e-07 -p -0.257187 0 0.445461 -n "%s%s_hexagon_CON" '''  %( name, (x+1) ) ) 
            
                cmds.setAttr ( hexagon_CON+'.rz', -90 )
                cmds.makeIdentity ( a=1, r=1 )
            
                hexagon_CONShape = cmds.listRelatives ( s=1 )[0]
                hexagon_CONShape = cmds.rename ( hexagon_CONShape, '%s%s_hexagon_CONShape' %( name, (x+1) ) )
                cmds.setAttr ( hexagon_CONShape+'.overrideEnabled', 1 )
                cmds.setAttr ( hexagon_CONShape+'.overrideColor', CON_color )
                self.hexagon_CON_list.append ( hexagon_CON )
    
                cmds.group ( n='%s%s_extra_NUL' % ( name, (x+1) ) )
                each_hexagon_NUL = cmds.group ( n='%s%s_NUL' % ( name, (x+1) ) )
                self.hexagon_NUL_list.append ( each_hexagon_NUL )    
                cmds.delete ( cmds.parentConstraint ( parent_list[x], each_hexagon_NUL ) )
                
                
                
            instance_hexagon_NUL_list = [] + self.hexagon_NUL_list
            instance_hexagon_CON_list = [] + self.hexagon_CON_list

            for x in range( len(parent_list)-1 ):
                cmds.parent ( instance_hexagon_NUL_list[-1], instance_hexagon_CON_list[-2] )
                del instance_hexagon_NUL_list[-1], instance_hexagon_CON_list[-1]





    ##########################################################################################################################################################################################################
    ## BFK Control Rigging 
    ##########################################################################################################################################################################################################

    ### BFK rig Pre ###  self.BFK_con_num, self.divided_con_list, self.position_list
    def BFK_control_pre( self, BFK_between_num, controler_list ):
            
        inatance_con_list = [] + controler_list
        
        self.divided_con_list = []
        
        while len(inatance_con_list) >= BFK_between_num:
            
            each_divided_con_list = inatance_con_list[0:BFK_between_num]

            self.divided_con_list.append ( each_divided_con_list )
            
            del inatance_con_list[0:BFK_between_num]
            
        if inatance_con_list:
           
            self.divided_con_list.append ( inatance_con_list )
        
        
        self.point_list = []    
            
        for x in range(len(self.divided_con_list)):
            each_BFK_positon_list = cmds.xform ( self.divided_con_list[x][0], q=1, rp=1, ws=1 )
            self.point_list.append( each_BFK_positon_list )
            
        
        self.cycle_num = len(self.divided_con_list)
            

            

    ### setting BFK controler ###
    def setting_BFK_controler( self, BFK_con_list, BFK_nul_list ):
        
        for x in range( self.cycle_num ):
            
            target_con_extra_nul = cmds.listRelatives ( self.divided_con_list[x][0], p=1 )[0]
            target_con_nul = cmds.listRelatives ( target_con_extra_nul, p=1 )[0]
            cmds.parent ( target_con_nul, BFK_con_list[x] )

            for y in self.divided_con_list[x]:
                each_target_con_extra_nul = cmds.listRelatives ( y, p=1 )[0]    
                cmds.connectAttr ( BFK_con_list[x]+'.rotate', each_target_con_extra_nul+'.rotate' )
                    
        
        instance_divided_con_list = [] + self.divided_con_list
 
        
        del instance_divided_con_list[-1]

        for x in range( self.cycle_num-1 ):
            
            cmds.parent ( BFK_nul_list[-1], instance_divided_con_list[-1][-1] )
           
            del self.hexagon_NUL_list[-1]
            del instance_divided_con_list[-1]
            
        
                  
            

    ##########################################################################################################################################################################################################
    ## IK-PATH Control Rigging 
    ##########################################################################################################################################################################################################
    
    ### 커브위에 패스 컨트롤러의 위치를 찾음 ###  self.motionPath_LOC_list, self.motionPath_LOCShape_list, self.path_curveShape
    def find_path_controler_position( self, path_con_num, sel_curve ):          
        
        position_list = []   
        motionPath_node_list = []
        self.motionPath_LOC_list = []
        self.motionPath_LOCShape_list = []

        for x in range( path_con_num ):
            each_motionPath_LOC = cmds.spaceLocator ( n='%s_motionPath%s_LOC' % ( self.splitName, (x+1) ) )[0]
            self.motionPath_LOC_list.append( each_motionPath_LOC )
            
            each_motionPath_LOCShape = cmds.listRelatives ( each_motionPath_LOC, s=1 )[0]
            self.motionPath_LOCShape_list.append( each_motionPath_LOCShape )
            
            each_motionPath_node = cmds.pathAnimation ( each_motionPath_LOC, stu=1, etu=10, c=sel_curve, f=1, fa='x', ua='y', wut='scene', fm=1 )
            motionPath_node_list.append( each_motionPath_node )
      
            
            cmds.setAttr ( each_motionPath_node+'.uValue', (1.0/(path_con_num-1))*x )
            
            each_position = cmds.getAttr ( each_motionPath_LOC+'.translate' )[0]
            position_list.append( each_position )
            
        
        path_curve = cmds.curve ( p=position_list, n='%s_path_CRV' %( self.splitName ) )
        self.path_curveShape = cmds.listRelatives ( path_curve, s=1 )[0]
        
        
        path_LOC_GRP = cmds.group ( self.motionPath_LOC_list, n=self.splitName+'_path_LOC_GRP' )
        cmds.setAttr ( path_LOC_GRP+'.visibility', 0 )
        cmds.parent ( path_LOC_GRP, path_curve, self.LineRig_GRP )
        
        cmds.delete ( motionPath_node_list )
        
        
	

	### 컨트롤러를 패스에 붙임 ###  self.posiotion_LOC_list
    def controler_follow_curve_pre( self, path_curveShape, controler_list ):
        
        self.posiotion_LOC_list = []
        
        for x in range( self.cycle_num ):
            each_NPC_node = cmds.createNode ( 'nearestPointOnCurve', n='%s%s_NPC' %( self.splitName, (x+1) ) )
            cmds.connectAttr ( path_curveShape+'.worldSpace[0]', each_NPC_node+'.inputCurve' )
        
            each_position_LOC = cmds.spaceLocator ( n='%s%s_position_LOC' %( self.splitName, (x+1) ) )[0]
            self.posiotion_LOC_list.append( each_position_LOC )
            each_position_LOCShape = cmds.listRelatives ( each_position_LOC, s=1 )[0]
            cmds.delete ( cmds.parentConstraint ( controler_list[x], each_position_LOC ) )

            controler_Position = cmds.getAttr ( each_position_LOCShape+'.worldPosition[0]' )[0]
            cmds.setAttr ( each_NPC_node+'.inPosition', controler_Position[0], controler_Position[1], controler_Position[2] )
            curve_position_parameter = cmds.getAttr ( each_NPC_node+'.parameter' )

            
            
            each_PCI_node = cmds.createNode ( 'pointOnCurveInfo', n='%s%s_PCI' %( self.splitName, (x+1) ) )
            cmds.connectAttr ( path_curveShape+'.worldSpace[0]',  each_PCI_node+'.inputCurve' )
            cmds.setAttr ( each_PCI_node+'.parameter', curve_position_parameter )
            
            cmds.connectAttr ( each_PCI_node+'.position', each_position_LOC+'.translate' )
            
        position_LOC_GRP = cmds.group ( self.posiotion_LOC_list, n='%s_position_locator_GRP' %( self.splitName ) )
        cmds.setAttr ( position_LOC_GRP+'.visibility', 0 )
        cmds.parent ( position_LOC_GRP, self.LineRig_GRP ) 
            





    ##########################################################################################################################################################################################################
    ## Setting Controler 
    ##########################################################################################################################################################################################################
    
    def setting_controler(self):
        
        controler_type_value = cmds.optionMenu ( 'controler_type', q=1, v=1 )
        
        if controler_type_value ==' FK':
            
            self.greate_controler( 'circle', 6, self.joint_list, self.splitName )    
                            
            self.sphere_CON_list = self.circle_CON_list
            cmds.parent ( self.circle_NUL_list[0], self.LineRig_GRP )
        

        elif controler_type_value==' IK':
            
            self.greate_controler( 'sphere', 13, self.joint_list, self.splitName )
            cmds.parent ( self.sphere_NUL_list, self.LineRig_GRP  )
            cmds.group ( self.sphere_NUL_list, n='%s_move_NUL_GRP ' % ( self.splitName ) )
            

        elif controler_type_value ==' FK - IK':
            
            self.greate_controler( 'circle', 6, self.joint_list, self.splitName )
            self.greate_controler( 'sphere', 13, self.joint_list, self.splitName )
            
            for x in range( self.cycle_num ):
                cmds.parent ( self.sphere_NUL_list[x], self.circle_CON_list[x] )
                
            cmds.parent ( self.circle_NUL_list[0], self.LineRig_GRP )


        elif controler_type_value ==' Path-IK - IK':
            
            get_path_con_num = int( cmds.textField ( self.setting_num_tf, q=1, tx=1 ) )
            
            self.find_path_controler_position( get_path_con_num, self.sel_curve )
            
            self.greate_controler( 'sphere', 14, self.motionPath_LOC_list, self.splitName+'_path' )
            cmds.parent ( self.sphere_NUL_list, self.LineRig_GRP  )
            cmds.group ( self.sphere_NUL_list, n='%s_path_move_NUL_GRP ' % ( self.splitName ) )
            
            for x in range( get_path_con_num ):
                cmds.parentConstraint ( self.sphere_CON_list[x], self.motionPath_LOC_list[x], mo=1 )
                cmds.connectAttr ( self.motionPath_LOCShape_list[x]+'.worldPosition', self.path_curveShape+'.controlPoints[%s]' %(x) )
                
            self.greate_controler( 'sphere', 13, self.joint_list, self.splitName )
            cmds.parent ( self.sphere_NUL_list, self.LineRig_GRP  )
            cmds.group ( self.sphere_NUL_list, n='%s_move_NUL_GRP ' % ( self.splitName ) )
            
            self.controler_follow_curve_pre( self.path_curveShape, self.sphere_CON_list )
            
            for x in range( self.cycle_num ):
                cmds.parentConstraint ( self.posiotion_LOC_list[x], self.sphere_NUL_list[x] )    
            
            self.controler_scale( self.sphere_CON_list, 0.6 )
            
            
        elif controler_type_value ==' Add B-FK':
            
            get_setting_num = int( cmds.textField ( self.setting_num_tf, q=1, tx=1 ) )
           
            self.controler_list = cmds.ls ( sl=1 )
            self.splitName = self.controler_list[0].split('_CON')[0]
            
            self.BFK_control_pre( get_setting_num, self.controler_list )
            
            self.joint_list = self.joint_generate( 'BFK' )
            
            self.greate_controler( 'hexagon', 17, self.joint_list, self.splitName+'_BFK' )
            self.controler_scale( self.hexagon_CON_list, 6.0 )
            cmds.delete ( self.joint_list )
        
            self.setting_BFK_controler( self.hexagon_CON_list, self.hexagon_NUL_list )
            
            
            


    ##########################################################################################################################################################################################################
    ## Hair Simulation OP
    ##########################################################################################################################################################################################################
            
    ### self.sel_curve 커브에 헤어 시뮬레이션 적용 ### self.baseCurve_shape, self.outputCurve_shape, self.follicle_node, self.hair_CON
    def apply_hair_SIM(self, setting_option ):
        
        cmds.select ( self.sel_curve )

        mel.eval ( 'makeCurvesDynamicHairs 0 0 1' )

        baseCurve = cmds.rename ( self.sel_curve, self.splitName+'_hair_base_CRV' )
        self.baseCurve_shape = cmds.listRelatives ( baseCurve, s=1 )[1]
        cmds.setAttr ( self.baseCurve_shape+'.overrideEnabled', 1 )
        cmds.setAttr ( self.baseCurve_shape+'.overrideColor', 16 )
        

        self.follicle_node = cmds.listRelatives ( baseCurve, p=1 )[0]
        follicleShape_node = cmds.listRelatives ( self.follicle_node, s=1 )[0]
        follicle_GRP = cmds.listRelatives ( self.follicle_node, p=1 )[0]
        outputCurve_pre = cmds.listConnections ( follicleShape_node+'.outCurve' )[0]
        outputCurve_GRP = cmds.listRelatives ( outputCurve_pre, p=1 )[0]
        outputCurve = cmds.rename ( outputCurve_pre, self.splitName+'_hair_output_CRV' )
        self.outputCurve_shape = cmds.listRelatives ( outputCurve, s=1 )[0]

        hairSystem_node = cmds.listConnections ( follicleShape_node+'.outHair' )[0]
        hairSystemShape_node = cmds.listRelatives ( hairSystem_node, s=1 )[0]
        nucleus_node = cmds.listConnections ( hairSystemShape_node+'.startState' )[0]
        cmds.setAttr ( nucleus_node+'.visibility', 0 )


        self.hair_CON = mel.eval ( '''curve -d 1 -p 2 2 2 -p 2 2 -2 -p -2 2 -2 -p -2 -2 -2 -p 2 -2 -2 -p 2 2 -2 -p -2 2 -2 -p -2 2 2 -p 2 2 2 -p 2 -2 2 -p 2 -2 -2 
                                            -p -2 -2 -2 -p -2 -2 2 -p 2 -2 2 -p -2 -2 2 -p -2 2 2 -k 0 -k 1 -k 2 -k 3 -k 4 -k 5 -k 6 -k 7 -k 8 -k 9 -k 10 -k 11 -k 
                                            12 -k 13 -k 14 -k 15 -n "%s_hair_dynamic_CON" ''' % ( self.splitName ) )
        
        self.hair_NUL = cmds.group ( n=self.splitName+'_hair_dynamic_NUL' )

        hair_CONShape = cmds.listRelatives ( self.hair_CON, s=1 )[0]
        cmds.setAttr ( hair_CONShape+'.overrideEnabled', 1 )
        cmds.setAttr ( hair_CONShape+'.overrideColor', 14 )

        
        item_list = [ 'tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'sx', 'sy', 'sz', 'visibility' ]

        for x in item_list:
            cmds.setAttr ( self.hair_CON+'.%s' %(x), l=1, k=0, cb=0 )

        cmds.addAttr ( self.hair_CON, ln='work_mode', at='enum', en='animation:simulation:', k=1 )
        cmds.setAttr ( self.hair_CON+'.work_mode', 1 )
        cmds.setDrivenKeyframe ( follicleShape_node+'.simulationMethod', cd=self.hair_CON+'.work_mode'  )
        cmds.setDrivenKeyframe ( hairSystemShape_node+'.simulationMethod', cd=self.hair_CON+'.work_mode' )
        cmds.setAttr ( self.hair_CON+'.work_mode', 0 )
        cmds.setAttr ( follicleShape_node+'.simulationMethod', 0 )
        cmds.setAttr ( hairSystemShape_node+'.simulationMethod', 1 )
        cmds.setDrivenKeyframe ( follicleShape_node+'.simulationMethod', cd=self.hair_CON+'.work_mode'  )
        cmds.setDrivenKeyframe ( hairSystemShape_node+'.simulationMethod', cd=self.hair_CON+'.work_mode' )
        cmds.setAttr ( self.hair_CON+'.work_mode', 1, k=0, cb=1 )

        hair_addAttr_tuple = { 'stiffness_scale' : 0.0, 'damp' : 0.5, 'guide_attract' : 0.01, 'bend' : 0.5 } 
        hair_setAttr_tuple = { nucleus_node+'.startFrame' : 950, follicleShape_node+'.pointLock' : 1, hairSystemShape_node+'.stiffnessScale[1].stiffnessScale_Interp' : 1, 
                               hairSystemShape_node+".stiffnessScale[1].stiffnessScale_Position" : 1, hairSystemShape_node+'.attractionScale[1].attractionScale_FloatValue' : 0.0 }
        hair_conAttr_tuple = { self.hair_CON+'.damp' : hairSystemShape_node+'.damp', self.hair_CON+'.bend' : hairSystemShape_node+'.bendResistance', 
                               self.hair_CON+'.stiffness_scale' : hairSystemShape_node+'.stiffnessScale[1].stiffnessScale_FloatValue', self.hair_CON+'.guide_attract' : hairSystemShape_node+'.startCurveAttract' }
        
        for x in range(4):
            cmds.addAttr ( self.hair_CON, ln=hair_addAttr_tuple.keys()[x], at='double', dv=hair_addAttr_tuple.values()[x], k=1 )
            
        for x in range(4):    
            cmds.setAttr ( hair_setAttr_tuple.keys()[x], hair_setAttr_tuple.values()[x] )
            cmds.connectAttr ( hair_conAttr_tuple.keys()[x], hair_conAttr_tuple.values()[x] )

        
        if setting_option=='IK':
            hair_IK = cmds.ikHandle ( sj=self.joint_list[0], ee=self.joint_list[-1], c=outputCurve, n=self.splitName+'_hair_dynamic_HDL', sol='ikSplineSolver', ccv=0, roc=0, pcv=0 )[0]
            cmds.parent ( hair_IK, self.LineRig_GRP )
            cmds.parentConstraint ( self.joint_list[-1], self.hair_NUL )
            
        elif setting_option=='TIP_CON':
            
            cmds.select ( outputCurve+'.cv[%s]' %( len(self.joint_list) ) )
            
            T_constraint_shape_node = mel.eval ( '''createNConstraint transform 0''' )[0]
            T_constraint_node = cmds.listRelatives ( T_constraint_shape_node, p=1 )
            
            self.greate_controler( 'sphere', 16, T_constraint_node, self.splitName+'_tip' )
            self.controler_scale( self.sphere_CON_list, 1.5 )
            
            cmds.parent ( T_constraint_node[0], self.sphere_NUL_list, self.LineRig_GRP )
            
            cmds.parentConstraint ( self.constraint_CON, self.follicle_node, mo=1 )
            cmds.parentConstraint ( self.sphere_CON_list[0], T_constraint_node[0], mo=1 )
            
            cmds.setAttr (  self.hair_CON+'.guide_attract', 0 )
            
            cmds.parentConstraint ( self.end_LOC, self.sphere_CON_list[0], mo=1 )
            
            
            
        cmds.parent ( outputCurve_GRP, hairSystem_node, self.hair_NUL, self.LineRig_GRP )



    ### self.LOC_NUL_list ###
    def curve_control_locator( self, curveShape ):
        
        self.LOC_NUL_list = []
        
        for x in range( self.cycle_num ):
            each_LOC = cmds.spaceLocator ( n=self.splitName+'hair%s_LOC' %( x ) )[0]
            each_LOCShape = cmds.listRelatives ( each_LOC, s=1 )[0]
            each_LOC_GRP = cmds.group ( n=self.splitName+'hair%s_locator_NUL' %( x ) )
            cmds.delete ( cmds.parentConstraint ( self.joint_list[x], each_LOC_GRP, mo=0 ) )
            cmds.connectAttr ( each_LOCShape+'.worldPosition', curveShape+'.controlPoints[%s]' %(x) )

            self.LOC_NUL_list.append(each_LOC_GRP)

            
        cmds.select ( self.LOC_NUL_list )
        self.LOC_NUL_GRP = cmds.group ( n=self.splitName+'_curve_locator_GRP' )
        cmds.setAttr ( self.LOC_NUL_GRP+'.visibility', 0 )
        
        cmds.parent ( self.LOC_NUL_GRP, self.LineRig_GRP )



    ### self.Skin_joint_list ###
    def apply_hair_SIM_advance(self):

        for x in range( self.cycle_num ):
            cmds.parentConstraint ( self.sphere_CON_list[x], self.LOC_NUL_list[x], mo=1 )
    
    
    def hair_SIM_TIP_CON_pre(self):
        
        self.end_LOC = cmds.spaceLocator ( n=self.splitName+'_end_LOC' )
        contraint_node = cmds.parentConstraint ( self.controler_list[-1], self.end_LOC, mo=0 )
            
        last_start_frame = cmds.playbackOptions( q=1, min=1 ) -1
        last_end_frame = cmds.playbackOptions( q=1, max=1 ) +1
        
        cmds.select ( self.end_LOC )
        cmds.bakeResults ( sm=1, t=( last_start_frame, last_end_frame), s=0 )
            
        cmds.delete ( contraint_node )
            
        cmds.parent ( self.end_LOC, self.LineRig_GRP )
           
            
    ##########################################################################################################################################################################################################            
    ## Up-Axis Control Rigging
    ##########################################################################################################################################################################################################
    
    ### self.Skin_joint_list, self.up_loc_list ###
    def upAxis_control_rig(self):
        
        self.Skin_joint_list = self.joint_generate( 'Skin' )
        
        cmds.parent ( self.Skin_joint_list[0], self.LineRig_GRP )        
        
        self.up_loc_list = []
        
        for x in range( self.cycle_num ):
            each_up_LOC = cmds.spaceLocator ( n='%s%s_up_LOC' % ( self.splitName, (x+1) ) )[0]
            self.up_loc_list.append(each_up_LOC)
            cmds.delete ( cmds.parentConstraint ( self.sphere_CON_list[x], each_up_LOC, mo=0 ) )
            cmds.parent ( each_up_LOC, self.sphere_CON_list[x] )
        
        aim_vector = cmds.spaceLocator ( n=self.splitName+'_aim_vector_LOC' )
        cmds.delete ( cmds.parentConstraint ( self.sphere_CON_list[-1], aim_vector, mo=0 ) )
        cmds.parent ( aim_vector, self.joint_list[-1] )
        cmds.select ( self.up_loc_list )
        cmds.move ( 0, 10, 0, r=1, os=1, wd=1 )
        cmds.move ( 8, 0, 0, aim_vector, r=1, os=1 )
         
        temp_joint_list = [] + self.joint_list
        del temp_joint_list[0]
        aim_parent_list = [] + temp_joint_list + aim_vector
        
        for x in range( self.cycle_num ):
            cmds.pointConstraint ( self.joint_list[x], self.Skin_joint_list[x], mo=1 )
            cmds.aimConstraint ( aim_parent_list[x], self.Skin_joint_list[x], mo=1, wut='object',  wuo=self.up_loc_list[x] )
            
            
            
    

    ##########################################################################################################################################################################################################
    ## 리깅 실행 함수    
    ##########################################################################################################################################################################################################
    
    def ringging_operate(self):
        
        rigging_type_value = cmds.optionMenu ( 'rigging_type', q=True, v=True ) 
        
        if rigging_type_value==' Controler':
            for x in range( self.cycle_num ):
                cmds.parentConstraint ( self.sphere_CON_list[x], self.joint_list[x], mo=1 )
        
        elif rigging_type_value==' Curve Bind':
            self.curve_control_locator( self.sel_curveShape )
            
            for x in range( self.cycle_num ):
                cmds.parentConstraint ( self.sphere_CON_list[x], self.LOC_NUL_list[x] )
                
        elif rigging_type_value==' Spline-IK':
            self.curve_control_locator( self.sel_curveShape )
            cmds.ikHandle ( sj=self.joint_list[0], ee=self.joint_list[-1], c=self.sel_curve, n=self.splitName+'_HDL', sol='ikSplineSolver', ccv=0, roc=0, pcv=0 )
            
            for x in range( self.cycle_num ):
                cmds.parentConstraint ( self.sphere_CON_list[x], self.LOC_NUL_list[x] )
                
            if cmds.checkBox ( self.upAxis_rig_option, q=1, v=1 ):
                self.upAxis_control_rig()
                
        elif rigging_type_value==' Hair Dynamic':
            self.apply_hair_SIM('IK' )
            self.curve_control_locator( self.baseCurve_shape )  
            self.apply_hair_SIM_advance()
            
            if cmds.checkBox ( self.upAxis_rig_option, q=1, v=1 ):
                self.upAxis_control_rig()
                
              



    ##########################################################################################################################################################################################################
    ## Instance Rigging 
    ##########################################################################################################################################################################################################
    
    ###인스턴스 리깅 pre ###  self.controler_list, self.constraint_CON
    def controler_pre(self):
        
        start_frame = cmds.playbackOptions( q=1, min=1 ) 
        cmds.currentTime ( start_frame, e=1 ) 
        
        self.controler_list = cmds.ls ( sl=1 )
        self.constraint_CON = self.controler_list[0]
        del self.controler_list[0]
        
        self.cycle_num = len(self.controler_list)

        con_splitName = self.controler_list[0].split('_CON')[0]
        
        if ':' in con_splitName:
            NS_con_splitName = con_splitName.split(':')[0]
            onlyName = con_splitName.split(':')[1]
            self.splitName = NS_con_splitName + '_' + onlyName
            
        else:
            self.splitName = con_splitName
        
        
        

    ### 선택한 컨트롤러에 커브를 생성 ###  self.point_list
    def generate_curve_on_controler(self):
        
        self.point_list = []
        
        for x in self.controler_list:
            con_each_xform = cmds.xform ( x, q=1, t=1, ws=1 )
            self.point_list.append( con_each_xform )
        
        self.base_curve = cmds.curve ( p=self.point_list, n=self.splitName+'_CRV' )
        cmds.select ( self.base_curve )
        
        
    

    ### self.controler_list와 self.LOC_NUL_list를 각각 컨스트레인 후 베이크 ###     
    def apply_hair_SIM_instance(self):
        
        delete_contraint_node_list = [] 
        
        last_start_frame = cmds.playbackOptions( q=1, min=1 ) -1
        last_end_frame = cmds.playbackOptions( q=1, max=1 ) +1
        
        cmds.setAttr ( self.hair_CON+'.work_mode', 0 )
        
        
        self.greate_controler( 'circle', 16, self.joint_list, self.splitName+'_instance' )
        self.controler_scale( self.circle_CON_list, 0.7 )
  
        ins_con_parent_NUL = cmds.group ( n=self.splitName+'ins_con_parent_NUL', em=1 )
        cmds.delete ( cmds.parentConstraint ( self.circle_NUL_list[0], ins_con_parent_NUL, mo=0 ) )
        cmds.parent ( self.circle_NUL_list[0], ins_con_parent_NUL )
        cmds.parent ( ins_con_parent_NUL, self.LineRig_GRP )
        
        cmds.parentConstraint ( self.constraint_CON, ins_con_parent_NUL, mo=1 )
        
        cmds.addAttr ( self.hair_CON, ln='instance_controler_vis', at='bool', k=1 )
        cmds.connectAttr ( self.hair_CON+'.instance_controler_vis', ins_con_parent_NUL+'.visibility' )


        for x in range( self.cycle_num ):
            each_contraint_node = cmds.parentConstraint ( self.controler_list[x], self.circle_CON_list[x], mo=1 )[0]
            cmds.parentConstraint ( self.circle_CON_list[x], self.LOC_NUL_list[x], mo=1 )
            delete_contraint_node_list.append( each_contraint_node )
            
        cmds.parentConstraint ( self.constraint_CON, self.joint_list[0], mo=1 )
        
        cmds.select ( self.circle_CON_list )
        

        cmds.bakeResults ( sm=1, t=( last_start_frame, last_end_frame), s=0 )
        
        cmds.delete ( delete_contraint_node_list )
        
            
    
    ### 컨트롤러와 Inatance Rig Controler를 컨스트레인 후 베이크 
    def instance_follow_controler( self, child_list,  parant_NUL ):
        
        constraint_node_list = []
        
        last_start_frame = cmds.playbackOptions( q=1, min=1 ) -1
        last_end_frame = cmds.playbackOptions( q=1, max=1 ) +1
        
        keyLOC_parent_NUL = cmds.group ( n=self.splitName+'key_LOC_parent_NUL', em=1 )
        cmds.delete ( cmds.parentConstraint ( parant_NUL, keyLOC_parent_NUL, mo=0 ) )
        cmds.parent ( parant_NUL, keyLOC_parent_NUL )
        cmds.parent ( keyLOC_parent_NUL, self.LineRig_GRP )
        
        cmds.parentConstraint ( self.constraint_CON, keyLOC_parent_NUL, mo=1 )
            
        
        for x in range( len( child_list ) ):
            each_constraint_node = cmds.parentConstraint ( self.controler_list[x], child_list[x] , mo=1 )[0]
            constraint_node_list.append(each_constraint_node)
            
        cmds.select ( child_list )
        
        cmds.bakeResults ( sm=1, t=( last_start_frame, last_end_frame), s=0 )
        
        cmds.delete ( constraint_node_list )

  

    ### 입력한 리스트와 self.controler_list를 각각 컨스트레인 ###
    def controler_follow_instance( self, parent_list, bake_CON, bake_type ):
        

        
        constraint_node_list = []
        
        if cmds.getAttr ( self.controler_list[0]+'.tx', l=1 )==0 and cmds.getAttr ( self.controler_list[0]+'.rx', l=1 )==0:
            
            for x in range( len( parent_list ) ):
                each_constraint_node = cmds.parentConstraint ( parent_list[x], self.controler_list[x] , mo=1 )[0]
                constraint_node_list.append(each_constraint_node)
                
        elif cmds.getAttr ( self.controler_list[0]+'.tx', l=1 )==0 and cmds.getAttr ( self.controler_list[0]+'.rx', l=1 )==1:
            
            for x in range( len( parent_list ) ):
                each_constraint_node = cmds.pointConstraint ( parent_list[x], self.controler_list[x] , mo=1 )[0]
                constraint_node_list.append(each_constraint_node)    
                
        elif cmds.getAttr ( self.controler_list[0]+'.tx', l=1 )==1 and cmds.getAttr ( self.controler_list[0]+'.rx', l=1 )==0:   
         
            for x in range( len( parent_list ) ):
                each_constraint_node = cmds.orientConstraint ( parent_list[x], self.controler_list[x] , mo=1 )[0]
                constraint_node_list.append(each_constraint_node)            
        
        delete_node_list = [ self.LineRig_GRP ]
        
        bake_contoler_list = [] + self.controler_list
        
        bake_type_tuple = { 'Default' : 0, 'Smart' : 1 }

        cmds.addAttr ( bake_CON, ln='bake_type', at='enum', en='Default:Smart:', k=1 )
        cmds.setAttr ( bake_CON+'.bake_type', bake_type_tuple[bake_type] )
        cmds.setAttr ( bake_CON+'.bake_type', k=0, cb=1, l=1 )
        
        cmds.addAttr ( bake_CON, ln='bake_controler_list', dt='stringArray' )
        cmds.setAttr ( bake_CON+'.bake_controler_list', type='stringArray', *( [len(bake_contoler_list)] + bake_contoler_list )  )
        
        cmds.addAttr ( bake_CON, ln='delete_node_list', dt='stringArray' )
        cmds.setAttr ( bake_CON+'.delete_node_list', type='stringArray', *( [len(delete_node_list)] + delete_node_list )  )
        
        instance_rigging_type_value = cmds.optionMenu ( 'instance_rigging_type', q=1, v=1 )
        
        if instance_rigging_type_value==' HairSIM Rig - Simple' or instance_rigging_type_value==' HairSIM Rig - Advance':
            cmds.setAttr ( self.hair_CON+'.work_mode', 1 )
        


        
        
    ### Controler Bake and Delete Node ###
    def bake_controler(self, *args):
        
        last_start_frame = cmds.playbackOptions( q=1, min=1 ) -1
        last_end_frame = cmds.playbackOptions( q=1, max=1 ) +1
        
        gather_bake_contoler_list = []
        gather_delete_node_list = []
        reprsent_controler_list = cmds.ls ( sl=1 )

        for x in reprsent_controler_list:
            each_bake_contoler_list = cmds.getAttr ( x+'.bake_controler_list' )
            gather_bake_contoler_list = gather_bake_contoler_list + each_bake_contoler_list
            
            each_delete_node_list = cmds.getAttr ( x+'.delete_node_list' )
            gather_delete_node_list = gather_delete_node_list + each_delete_node_list          
            
        
        get_bake_type = cmds.getAttr ( reprsent_controler_list[0]+'.bake_type' )
        
        cmds.select ( gather_bake_contoler_list  )


        if get_bake_type==1:
            cmds.bakeResults ( sm=1, t=( last_start_frame, last_end_frame), s=0, sr=1 )
        
        else:
            cmds.bakeResults ( sm=1, t=( last_start_frame, last_end_frame), s=0 )                
        

        cmds.delete ( gather_delete_node_list )
        
       
        


    ##########################################################################################################################################################################################################
    ## Rigging Operate Def 
    ##########################################################################################################################################################################################################

    def rig_OP(self, *args):
        
        controler_type_value = cmds.optionMenu ( 'controler_type', q=1, v=1 )
         
        if controler_type_value==' Add B-FK':
            self.setting_controler()
        else:
            self.jointCurve()
            self.setting_controler() 
            self.ringging_operate()
        



    def intance_rig_OP(self, *args):
        
        select_list = cmds.ls ( sl=1 )
        
        instance_rigging_type_value = cmds.optionMenu ( 'instance_rigging_type', q=True, v=True ) 
        
        if instance_rigging_type_value ==' FK Rig':
            
            self.controler_pre()
            self.LineRig_GRP = cmds.group ( n=self.splitName+'_LineRig_node_GRP', em=1 )
            self.greate_controler( 'circle', 30, self.controler_list, self.splitName )    
            
            self.instance_follow_controler( self.circle_NUL_list, self.circle_NUL_list[0] )
            self.controler_follow_instance( self.circle_CON_list, self.circle_CON_list[-1], 'Default' )
            cmds.parentConstraint ( self.constraint_CON, circle_parent_NUL, mo=1 )
            

        elif instance_rigging_type_value ==' FK Rig - Adance':
            
            self.controler_pre()
            self.generate_curve_on_controler()            
            self.jointCurve()
            self.greate_controler( 'circle', 30, self.joint_list, self.splitName )

            cmds.parent ( self.circle_NUL_list[0], self.LineRig_GRP )
            
            get_setting_num = int( cmds.textField ( self.IT_setting_num_tf, q=1, tx=1 ) )
            
            cmds.select ( self.circle_CON_list )
            self.BFK_control_pre( get_setting_num, self.circle_CON_list )

            self.splitName = self.circle_CON_list[0].split('_CON')[0]
            self.joint_list = self.joint_generate( 'BFK' )
            
            self.greate_controler( 'hexagon', 31, self.joint_list, self.splitName+'_BFK' )
            self.controler_scale( self.hexagon_CON_list, 6.0 )
            
            cmds.delete ( self.joint_list )
                
            self.setting_BFK_controler( self.hexagon_CON_list, self.hexagon_NUL_list  ) 
            cmds.parent ( self.hexagon_NUL_list[0], self.LineRig_GRP )        
            
            self.instance_follow_controler( self.circle_NUL_list, self.hexagon_NUL_list[0] )
            self.controler_follow_instance( self.circle_CON_list, self.circle_CON_list[-1], 'Default' )   
             

        else:
            
            if len(select_list) < 5:
                cmds.confirmDialog( title='Guide Massage', message='=== 컨트롤러를 5개 이상 선택해주세요 ! ===      ', button='OK' )
            
            else:               
        
                if instance_rigging_type_value == ' Path-IK Rig':
                	
                    self.controler_pre()
                    self.generate_curve_on_controler()
					
                    self.LineRig_GRP = cmds.group ( n=self.splitName+'_LineRig_node_GRP', em=1 )
					
                    get_path_con_num = int( cmds.textField ( self.IT_setting_num_tf, q=1, tx=1 ) )
                    self.find_path_controler_position( get_path_con_num, self.base_curve )
                    self.greate_controler( 'sphere', 30, self.motionPath_LOC_list, self.splitName+'_path' )
                    cmds.parent ( self.sphere_NUL_list, self.LineRig_GRP  )
                    
                    for x in range( get_path_con_num ):
                        cmds.parentConstraint ( self.sphere_CON_list[x], self.motionPath_LOC_list[x], mo=1 )
                        cmds.connectAttr ( self.motionPath_LOCShape_list[x]+'.worldPosition', self.path_curveShape+'.controlPoints[%s]' %(x) )
                        
                    self.controler_scale( self.sphere_CON_list, 1.4 )

                    self.controler_follow_curve_pre( self.path_curveShape, self.controler_list )
                    self.controler_follow_instance( self.posiotion_LOC_list, self.sphere_CON_list[-1], 'Default' )
                    
                    cmds.parentConstraint ( self.constraint_CON, self.sphere_NUL_list[0], mo=1 )
                

                elif instance_rigging_type_value ==' HairSIM Rig - Simple':

                    self.controler_pre()
                    self.generate_curve_on_controler()
                    self.jointCurve()
                    self.apply_hair_SIM( 'IK' )
                    self.curve_control_locator( self.baseCurve_shape ) 
                    self.controler_follow_instance( self.joint_list, self.hair_CON, 'Default' )
                    
                    cmds.parentConstraint ( self.constraint_CON, self.LOC_NUL_GRP, mo=1 )
                    cmds.parentConstraint ( self.constraint_CON, self.joint_list[0], mo=1 )
                    

                elif instance_rigging_type_value==' HairSIM Rig - Tip Con':
                
                    self.controler_pre()
                    self.generate_curve_on_controler()
                    self.jointCurve()
                    self.hair_SIM_TIP_CON_pre()
                    self.apply_hair_SIM( 'TIP_CON' )
                    self.controler_follow_curve_pre( self.outputCurve_shape, self.controler_list )
                   
                    for x in range(len(self.controler_list)):
                        cmds.parentConstraint ( self.posiotion_LOC_list[x], self.controler_list[x] )
                        
                    cmds.parentConstraint ( self.posiotion_LOC_list[-1], self.hair_NUL, mo=0 )
                    
                    cmds.select ( self.joint_list[0], self.follicle_node )
                    cmds.HideSelectedObjects ()
                        
                                        
                elif instance_rigging_type_value ==' HairSIM Rig - Advance':

                    self.controler_pre()
                    self.generate_curve_on_controler()
                    self.jointCurve()
                    self.apply_hair_SIM( 'IK' )
                    self.curve_control_locator( self.baseCurve_shape )
                    self.apply_hair_SIM_instance()
                    self.controler_follow_instance( self.joint_list, self.hair_CON, 'Default' )


  


def createWin():
    mykey = class_NINE()
    
createWin()