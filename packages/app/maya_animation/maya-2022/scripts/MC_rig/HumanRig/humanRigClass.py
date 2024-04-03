import maya.cmds as cmds



class human_rig:
    
    def __init__(self):

        self.skinJNT_list = [
                                #Body skin joints
                                'C_Skin_hip_JNT', 'C_Skin_spine1_JNT', 'C_Skin_spine2_JNT', 'C_Skin_spine3_JNT', 'C_Skin_chest_JNT', 'C_Skin_neck_JNT', 'C_Skin_neckTwist_JNT', 'C_Skin_neckTwist1_JNT', 'C_Skin_neckTwist2_JNT', 'C_Skin_head_JNT', 

                                #Right arm skin joints
                                'R_Skin_shoulder_JNT', 'R_Skin_upArm_JNT'       , 'R_Skin_upArmTwist_JNT'   , 'R_Skin_upArmTwist1_JNT'  , 'R_Skin_upArmTwist2_JNT'  , 'R_Skin_upArmTwist3_JNT'  , 'R_Skin_upArmTwist4_JNT', 
                                'R_Skin_foreArm_JNT' , 'R_Skin_foreArmTwist_JNT', 'R_Skin_foreArmTwist1_JNT', 'R_Skin_foreArmTwist2_JNT', 'R_Skin_foreArmTwist3_JNT', 'R_Skin_foreArmTwist4_JNT', 'R_Skin_hand_JNT', 
                                
                                #Right finger skin joints
                                'R_Skin_indexPalm_JNT' , 'R_Skin_index1_JNT' , 'R_Skin_index2_JNT' , 'R_Skin_index3_JNT' , 'R_Skin_index4_JNT', 
                                'R_Skin_middlePalm_JNT', 'R_Skin_middle1_JNT', 'R_Skin_middle2_JNT', 'R_Skin_middle3_JNT', 'R_Skin_middle4_JNT', 
                                'R_Skin_ringPalm_JNT'  , 'R_Skin_ring1_JNT'  , 'R_Skin_ring2_JNT'  , 'R_Skin_ring3_JNT'  , 'R_Skin_ring4_JNT', 
                                'R_Skin_pinkyPalm_JNT' , 'R_Skin_pinky1_JNT' , 'R_Skin_pinky2_JNT' , 'R_Skin_pinky3_JNT' , 'R_Skin_pinky4_JNT', 
                                'R_Skin_thumb1_JNT'    , 'R_Skin_thumb2_JNT' , 'R_Skin_thumb3_JNT' , 'R_Skin_thumb4_JNT' , 
                                
                                #Left arm skin joints
                                'L_Skin_shoulder_JNT', 'L_Skin_upArm_JNT'       , 'L_Skin_upArmTwist_JNT'   , 'L_Skin_upArmTwist1_JNT'  , 'L_Skin_upArmTwist2_JNT'  , 'L_Skin_upArmTwist3_JNT'  , 'L_Skin_upArmTwist4_JNT', 
                                'L_Skin_foreArm_JNT' , 'L_Skin_foreArmTwist_JNT', 'L_Skin_foreArmTwist1_JNT', 'L_Skin_foreArmTwist2_JNT', 'L_Skin_foreArmTwist3_JNT', 'L_Skin_foreArmTwist4_JNT', 'L_Skin_hand_JNT', 
                                
                                #Left finger skin joints
                                'L_Skin_indexPalm_JNT' , 'L_Skin_index1_JNT' , 'L_Skin_index2_JNT' , 'L_Skin_index3_JNT' , 'L_Skin_index4_JNT', 
                                'L_Skin_middlePalm_JNT', 'L_Skin_middle1_JNT', 'L_Skin_middle2_JNT', 'L_Skin_middle3_JNT', 'L_Skin_middle4_JNT', 
                                'L_Skin_ringPalm_JNT'  , 'L_Skin_ring1_JNT'  , 'L_Skin_ring2_JNT'  , 'L_Skin_ring3_JNT'  , 'L_Skin_ring4_JNT', 
                                'L_Skin_pinkyPalm_JNT' , 'L_Skin_pinky1_JNT' , 'L_Skin_pinky2_JNT' , 'L_Skin_pinky3_JNT' , 'L_Skin_pinky4_JNT', 
                                'L_Skin_thumb1_JNT'    , 'L_Skin_thumb2_JNT' , 'L_Skin_thumb3_JNT' , 'L_Skin_thumb4_JNT' ,  
                                
                                #Right leg skin joints
                                'R_Skin_leg_JNT'   , 'R_Skin_legTwist_JNT'   , 'R_Skin_legTwist1_JNT'   , 'R_Skin_legTwist2_JNT'   , 'R_Skin_legTwist3_JNT'   , 'R_Skin_legTwist4_JNT', 
                                'R_Skin_lowLeg_JNT', 'R_Skin_lowLegTwist_JNT', 'R_Skin_lowLegTwist1_JNT', 'R_Skin_lowLegTwist2_JNT', 'R_Skin_lowLegTwist3_JNT', 'R_Skin_lowLegTwist4_JNT', 
                                'R_Skin_foot_JNT'  , 'R_Skin_ball_JNT'       , 'R_Skin_toe_JNT', 
                                
                                #Left leg skin joints
                                'L_Skin_leg_JNT'   , 'L_Skin_legTwist_JNT'   , 'L_Skin_legTwist1_JNT'   , 'L_Skin_legTwist2_JNT'   , 'L_Skin_legTwist3_JNT'   , 'L_Skin_legTwist4_JNT', 
                                'L_Skin_lowLeg_JNT', 'L_Skin_lowLegTwist_JNT', 'L_Skin_lowLegTwist1_JNT', 'L_Skin_lowLegTwist2_JNT', 'L_Skin_lowLegTwist3_JNT', 'L_Skin_lowLegTwist4_JNT', 
                                'L_Skin_foot_JNT'  , 'L_Skin_ball_JNT'       , 'L_Skin_toe_JNT'


                              ]



        self.skin_HIK_hierachyList = {

                    #body joint hierachy
                    'C_Skin_hip_JNT':'Hips', 'C_Skin_spine1_JNT':'Spine', 'C_Skin_spine2_JNT':'Spine1', 'C_Skin_spine3_JNT':'Spine2','C_Skin_chest_JNT':'Spine3',

                    #neck joint hierachy
                    'C_Skin_neck_JNT':'Neck', 'C_Skin_head_JNT':'Head','C_Skin_neckTwist2_JNT':'Head','C_Skin_neckTwist1_JNT':'Neck','C_Skin_neckTwist_JNT':'Neck',

                    #L_arm joint hierachy
                    'L_Skin_shoulder_JNT':'LeftShoulder','L_Skin_upArm_JNT':'LeftArm','L_Skin_upArmTwist_JNT':'LeftArm', 'L_Skin_upArmTwist1_JNT':'LeftArm',  'L_Skin_upArmTwist2_JNT':'LeftArm', 'L_Skin_upArmTwist3_JNT':'LeftArm', 'L_Skin_upArmTwist4_JNT':'LeftArm', 
                    'L_Skin_foreArm_JNT':'LeftForeArm', 'L_Skin_foreArmTwist_JNT':'LeftForeArm', 'L_Skin_foreArmTwist1_JNT':'LeftForeArm', 'L_Skin_foreArmTwist2_JNT':'LeftForeArm', 'L_Skin_foreArmTwist3_JNT':'LeftForeArm','L_Skin_foreArmTwist4_JNT':'LeftForeArm',
                    'L_Skin_hand_JNT':'LeftHand','L_Skin_middle1_JNT':'LeftFingerBase',


                    #R_arm joint hierachy
                    'R_Skin_shoulder_JNT':'RightShoulder','R_Skin_upArm_JNT':'RightArm', 'R_Skin_upArmTwist_JNT':'RightArm', 'R_Skin_upArmTwist1_JNT':'RightArm',  'R_Skin_upArmTwist2_JNT':'RightArm', 'R_Skin_upArmTwist3_JNT':'RightArm', 'R_Skin_upArmTwist4_JNT':'RightArm', 
                    'R_Skin_foreArm_JNT':'RightForeArm', 'R_Skin_foreArmTwist_JNT':'RightForeArm', 'R_Skin_foreArmTwist1_JNT':'RightForeArm', 'R_Skin_foreArmTwist2_JNT':'RightForeArm', 'R_Skin_foreArmTwist3_JNT':'RightForeArm','R_Skin_foreArmTwist4_JNT':'RightForeArm',
                    'R_Skin_hand_JNT':'RightHand','R_Skin_middle1_JNT':'RightFingerBase',


                    #L_leg_joint_hierachy
                    'L_Skin_leg_JNT':'LeftUpLeg','L_Skin_legTwist_JNT':'LeftUpLeg', 'L_Skin_legTwist1_JNT':'LeftUpLeg','L_Skin_legTwist2_JNT':'LeftUpLeg','L_Skin_legTwist3_JNT':'LeftUpLeg','L_Skin_legTwist4_JNT':'LeftUpLeg',
                    'L_Skin_lowLeg_JNT':'LeftLeg','L_Skin_lowLegTwist_JNT':'LeftLeg', 'L_Skin_lowLegTwist1_JNT':'LeftLeg', 'L_Skin_lowLegTwist2_JNT':'LeftLeg', 'L_Skin_lowLegTwist3_JNT':'LeftLeg', 'L_Skin_lowLegTwist4_JNT':'LeftLeg',
                    'L_Skin_ball_JNT':'LeftToeBase', 'L_Skin_foot_JNT':'LeftFoot', 'L_Skin_foot_JNT':'LeftFoot', 'L_Skin_ball_JNT':'LeftToeBase',


                    #R_leg_joint_hierachy
                    'R_Skin_leg_JNT':'RightUpLeg','R_Skin_legTwist_JNT':'RightUpLeg', 'R_Skin_legTwist1_JNT':'RightUpLeg','R_Skin_legTwist2_JNT':'RightUpLeg','R_Skin_legTwist3_JNT':'RightUpLeg','R_Skin_legTwist4_JNT':'RightUpLeg',
                    'R_Skin_lowLeg_JNT':'RightLeg','R_Skin_lowLegTwist_JNT':'RightLeg', 'R_Skin_lowLegTwist1_JNT':'RightLeg', 'R_Skin_lowLegTwist2_JNT':'RightLeg', 'R_Skin_lowLegTwist3_JNT':'RightLeg', 'R_Skin_lowLegTwist4_JNT':'RightLeg',
                    'R_Skin_ball_JNT':'RightToeBase','R_Skin_foot_JNT':'RightFoot','R_Skin_foot_JNT':'RightFoot', 'R_Skin_ball_JNT':'RightToeBase',


                    #L finger joint hierachy
                    'L_Skin_thumb3_JNT':'L_Skin_thumb2_JNT', 'L_Skin_thumb2_JNT':'L_Skin_thumb1_JNT', 'L_Skin_thumb1_JNT':'LeftHand', 
                    'L_Skin_index3_JNT':'L_Skin_index2_JNT', 'L_Skin_index2_JNT':'L_Skin_index1_JNT', 'L_Skin_index1_JNT':'L_Skin_indexPalm_JNT','L_Skin_indexPalm_JNT':'LeftHand',
                    'L_Skin_middle3_JNT':'L_Skin_middle2_JNT', 'L_Skin_middle2_JNT':'L_Skin_middle1_JNT', 'L_Skin_middle1_JNT':'L_Skin_middlePalm_JNT','L_Skin_middlePalm_JNT':'LeftHand',
                    'L_Skin_ring3_JNT':'L_Skin_ring2_JNT', 'L_Skin_ring2_JNT':'L_Skin_ring1_JNT', 'L_Skin_ring1_JNT':'L_Skin_ringPalm_JNT','L_Skin_ringPalm_JNT':'LeftHand',
                    'L_Skin_pinky3_JNT':'L_Skin_pinky2_JNT', 'L_Skin_pinky2_JNT':'L_Skin_pinky1_JNT','L_Skin_pinky1_JNT':'L_Skin_pinkyPalm_JNT','L_Skin_pinkyPalm_JNT':'LeftHand',


                    #R finger joint hierachy
                    'R_Skin_thumb3_JNT':'R_Skin_thumb2_JNT', 'R_Skin_thumb2_JNT':'R_Skin_thumb1_JNT', 'R_Skin_thumb1_JNT':'RightHand', 
                    'R_Skin_index3_JNT':'R_Skin_index2_JNT', 'R_Skin_index2_JNT':'R_Skin_index1_JNT', 'R_Skin_index1_JNT':'R_Skin_indexPalm_JNT', 'R_Skin_indexPalm_JNT':'RightHand',
                    'R_Skin_middle3_JNT':'R_Skin_middle2_JNT', 'R_Skin_middle2_JNT':'R_Skin_middle1_JNT', 'R_Skin_middle1_JNT':'R_Skin_middlePalm_JNT','R_Skin_middlePalm_JNT':'RightHand',
                    'R_Skin_ring3_JNT':'R_Skin_ring2_JNT', 'R_Skin_ring2_JNT':'R_Skin_ring1_JNT', 'R_Skin_ring1_JNT':'R_Skin_ringPalm_JNT','R_Skin_ringPalm_JNT':'RightHand',
                    'R_Skin_pinky3_JNT':'R_Skin_pinky2_JNT', 'R_Skin_pinky2_JNT':'R_Skin_pinky1_JNT','R_Skin_pinky1_JNT':'R_Skin_pinkyPalm_JNT','R_Skin_pinkyPalm_JNT':'RightHand'

                    }


        self.fingerJoint_list = [
                                 'L_Skin_indexPalm_JNT', 'L_Skin_index1_JNT', 'L_Skin_index2_JNT', 'L_Skin_index3_JNT', 'L_Skin_index4_JNT', 'L_Skin_middlePalm_JNT', 'L_Skin_middle1_JNT', 'L_Skin_middle2_JNT', 'L_Skin_middle3_JNT', 'L_Skin_middle4_JNT', 'L_Skin_ringPalm_JNT', 'L_Skin_ring1_JNT', 'L_Skin_ring2_JNT', 'L_Skin_ring3_JNT', 'L_Skin_ring4_JNT', 'L_Skin_pinkyPalm_JNT', 'L_Skin_pinky1_JNT', 'L_Skin_pinky2_JNT', 'L_Skin_pinky3_JNT', 'L_Skin_pinky4_JNT', 'L_Skin_thumb1_JNT', 'L_Skin_thumb2_JNT', 'L_Skin_thumb3_JNT', 'L_Skin_thumb4_JNT',
                                 'R_Skin_indexPalm_JNT', 'R_Skin_index1_JNT', 'R_Skin_index2_JNT', 'R_Skin_index3_JNT', 'R_Skin_index4_JNT', 'R_Skin_middlePalm_JNT', 'R_Skin_middle1_JNT', 'R_Skin_middle2_JNT', 'R_Skin_middle3_JNT', 'R_Skin_middle4_JNT', 'R_Skin_ringPalm_JNT', 'R_Skin_ring1_JNT', 'R_Skin_ring2_JNT', 'R_Skin_ring3_JNT', 'R_Skin_ring4_JNT', 'R_Skin_pinkyPalm_JNT', 'R_Skin_pinky1_JNT', 'R_Skin_pinky2_JNT', 'R_Skin_pinky3_JNT', 'R_Skin_pinky4_JNT', 'R_Skin_thumb1_JNT', 'R_Skin_thumb2_JNT', 'R_Skin_thumb3_JNT', 'R_Skin_thumb4_JNT',
                                ]

        self.HIK_fingerJoint_list = [ 
                                 'LeftIndexPalm',   'LeftIndex1',  'LeftIndex2',  'LeftIndex3', 'LeftIndex4',   'LeftMiddlePalm', 'LeftMiddle1',  'LeftMiddle2',  'LeftMiddle3',  'LeftMiddle4',   'LeftRingPalm',  'LeftRing1',  'LeftRing2',  'LeftRing3',  'LeftRing4', 'LeftPinkyPalm',  'LeftPinky1',  'LeftPinky2',  'LeftPinky3',  'LeftPinky4',  'LeftThumb1',  'LeftThumb2',  'LeftThumb3',  'LeftThumb4',
                                 'RightIndexPalm', 'RightIndex1', 'RightIndex2', 'RightIndex3', 'RightIndex4', 'RightMiddlePalm', 'RightMiddle1', 'RightMiddle2', 'RightMiddle3', 'RightMiddle4', 'RightRingPalm', 'RightRing1', 'RightRing2', 'RightRing3', 'RightRing4', 'RightPinkyPalm', 'RightPinky1', 'RightPinky2', 'RightPinky3', 'RightPinky4', 'RightThumb1', 'RightThumb2', 'RightThumb3', 'RightThumb4',
                                ]

        self.fingerControler_list = ['R_FK_thumb3_CON', 'R_FK_thumb2_CON', 'R_FK_thumb1_CON', 'R_FK_pinky3_CON', 'R_FK_pinky2_CON', 'R_FK_pinky1_CON', 'R_FK_pinkyPalm_CON', 'R_FK_ring3_CON', 'R_FK_ring2_CON', 'R_FK_ring1_CON', 'R_FK_ringPalm_CON', 'R_FK_middle3_CON', 'R_FK_middle2_CON', 'R_FK_middle1_CON', 'R_FK_middlePalm_CON', 'R_FK_index3_CON', 'R_FK_index2_CON', 'R_FK_index1_CON', 'R_FK_indexPalm_CON', 'L_FK_index1_CON', 'L_FK_indexPalm_CON', 'L_FK_thumb3_CON', 'L_FK_thumb2_CON', 'L_FK_thumb1_CON', 'L_FK_pinky3_CON', 'L_FK_pinky2_CON', 'L_FK_pinky1_CON', 'L_FK_pinkyPalm_CON', 'L_FK_ring3_CON', 'L_FK_ring2_CON', 'L_FK_ring1_CON', 'L_FK_ringPalm_CON', 'L_FK_middle3_CON', 'L_FK_middle2_CON', 'L_FK_middle1_CON', 'L_FK_middlePalm_CON', 'L_FK_index3_CON', 'L_FK_index2_CON']







        self.connectSkinJoint_list = ['C_Skin_hip_JNT', 'C_Skin_spine1_JNT', 'C_Skin_spine2_JNT', 'C_Skin_spine3_JNT','C_Skin_chest_JNT', 'C_Skin_neck_JNT', 'C_Skin_head_JNT', 'L_Skin_leg_JNT', 'L_Skin_lowLeg_JNT', 'L_Skin_ball_JNT', 'L_Skin_foot_JNT', 'R_Skin_leg_JNT', 'R_Skin_lowLeg_JNT', 'R_Skin_ball_JNT', 'R_Skin_foot_JNT']   #'L_Skin_foot_JNT','R_Skin_foot_JNT',
        self.HIKJoint_list = [             'Hips',           'Spine',             'Spine1',            'Spine2',          'Spine3',            'Neck',              'Head',        'LeftUpLeg',       'LeftLeg',        'LeftToeBase',       'LeftFoot',      'RightUpLeg',       'RightLeg',        'RightToeBase',     'RightFoot']

        self.L_SkinArmJoint_list = ['L_Skin_shoulder_JNT','L_Skin_upArm_JNT', 'L_Skin_foreArm_JNT', 'L_Skin_hand_JNT', 'L_Skin_middle1_JNT']
        self.HIK_L_ArmJoint_list = [   'LeftShoulder',         'LeftArm',        'LeftForeArm',        'LeftHand',       'LeftFingerBase']
    
        self.R_SkinArmJoint_list = ['R_Skin_shoulder_JNT','R_Skin_upArm_JNT', 'R_Skin_foreArm_JNT', 'R_Skin_hand_JNT', 'R_Skin_middle1_JNT']
        self.HIK_R_ArmJoint_list = [   'RightShoulder',         'RightArm',        'RightForeArm',      'RightHand',      'RightFingerBase']

        ################################################################################################################################################################################################################################################################





        self.skin_crd_hierachyList = {

                    #body joint hierachy
                    'C_Skin_hip_JNT':'Crw_Hips', 'C_Skin_spine1_JNT':'Crw_Spine', 'C_Skin_spine2_JNT':'Crw_Spine1', 'C_Skin_spine3_JNT':'Crw_Spine2','C_Skin_chest_JNT':'Crw_Spine3',

                    #neck joint hierachy
                    'C_Skin_neck_JNT':'Crw_Neck', 'C_Skin_head_JNT':'Crw_Head','C_Skin_neckTwist2_JNT':'Crw_Head','C_Skin_neckTwist1_JNT':'Crw_Neck','C_Skin_neckTwist_JNT':'Crw_Neck',

                    #L_arm joint hierachy
                    'L_Skin_shoulder_JNT':'Crw_LeftShoulder','L_Skin_upArm_JNT':'Crw_LeftArm','L_Skin_upArmTwist_JNT':'Crw_LeftArm', 'L_Skin_upArmTwist1_JNT':'Crw_LeftArm',  'L_Skin_upArmTwist2_JNT':'Crw_LeftArm', 'L_Skin_upArmTwist3_JNT':'Crw_LeftArm', 'L_Skin_upArmTwist4_JNT':'Crw_LeftArm', 
                    'L_Skin_foreArm_JNT':'Crw_LeftForeArm', 'L_Skin_foreArmTwist_JNT':'Crw_LeftForeArm', 'L_Skin_foreArmTwist1_JNT':'Crw_LeftForeArm', 'L_Skin_foreArmTwist2_JNT':'Crw_LeftForeArm', 'L_Skin_foreArmTwist3_JNT':'Crw_LeftForeArm','L_Skin_foreArmTwist4_JNT':'Crw_LeftForeArm',
                    'L_Skin_hand_JNT':'Crw_LeftHand','L_Skin_middle1_JNT':'Crw_LeftFingerBase',


                    #R_arm joint hierachy
                    'R_Skin_shoulder_JNT':'Crw_RightShoulder','R_Skin_upArm_JNT':'Crw_RightArm', 'R_Skin_upArmTwist_JNT':'Crw_RightArm', 'R_Skin_upArmTwist1_JNT':'Crw_RightArm',  'R_Skin_upArmTwist2_JNT':'Crw_RightArm', 'R_Skin_upArmTwist3_JNT':'Crw_RightArm', 'R_Skin_upArmTwist4_JNT':'Crw_RightArm', 
                    'R_Skin_foreArm_JNT':'Crw_RightForeArm', 'R_Skin_foreArmTwist_JNT':'Crw_RightForeArm', 'R_Skin_foreArmTwist1_JNT':'Crw_RightForeArm', 'R_Skin_foreArmTwist2_JNT':'Crw_RightForeArm', 'R_Skin_foreArmTwist3_JNT':'Crw_RightForeArm','R_Skin_foreArmTwist4_JNT':'Crw_RightForeArm',
                    'R_Skin_hand_JNT':'Crw_RightHand','R_Skin_middle1_JNT':'Crw_RightFingerBase',


                    #L_leg_joint_hierachy
                    'L_Skin_leg_JNT':'Crw_LeftUpLeg','L_Skin_legTwist_JNT':'Crw_LeftUpLeg', 'L_Skin_legTwist1_JNT':'Crw_LeftUpLeg','L_Skin_legTwist2_JNT':'Crw_LeftUpLeg','L_Skin_legTwist3_JNT':'Crw_LeftUpLeg','L_Skin_legTwist4_JNT':'Crw_LeftUpLeg',
                    'L_Skin_lowLeg_JNT':'Crw_LeftLeg','L_Skin_lowLegTwist_JNT':'Crw_LeftLeg', 'L_Skin_lowLegTwist1_JNT':'Crw_LeftLeg', 'L_Skin_lowLegTwist2_JNT':'Crw_LeftLeg', 'L_Skin_lowLegTwist3_JNT':'Crw_LeftLeg', 'L_Skin_lowLegTwist4_JNT':'Crw_LeftLeg',
                    'L_Skin_ball_JNT':'Crw_LeftToeBase', 'L_Skin_foot_JNT':'Crw_LeftFoot', 'L_Skin_foot_JNT':'Crw_LeftFoot', 'L_Skin_ball_JNT':'Crw_LeftToeBase',


                    #R_leg_joint_hierachy
                    'R_Skin_leg_JNT':'Crw_RightUpLeg','R_Skin_legTwist_JNT':'Crw_RightUpLeg', 'R_Skin_legTwist1_JNT':'Crw_RightUpLeg','R_Skin_legTwist2_JNT':'Crw_RightUpLeg','R_Skin_legTwist3_JNT':'Crw_RightUpLeg','R_Skin_legTwist4_JNT':'Crw_RightUpLeg',
                    'R_Skin_lowLeg_JNT':'Crw_RightLeg','R_Skin_lowLegTwist_JNT':'Crw_RightLeg', 'R_Skin_lowLegTwist1_JNT':'Crw_RightLeg', 'R_Skin_lowLegTwist2_JNT':'Crw_RightLeg', 'R_Skin_lowLegTwist3_JNT':'Crw_RightLeg', 'R_Skin_lowLegTwist4_JNT':'Crw_RightLeg',
                    'R_Skin_ball_JNT':'Crw_RightToeBase','R_Skin_foot_JNT':'Crw_RightFoot','R_Skin_foot_JNT':'Crw_RightFoot', 'R_Skin_ball_JNT':'Crw_RightToeBase',


                    #L finger joint hierachy
                    'L_Skin_thumb3_JNT':'L_Skin_thumb2_JNT', 'L_Skin_thumb2_JNT':'L_Skin_thumb1_JNT', 'L_Skin_thumb1_JNT':'Crw_LeftHand', 
                    'L_Skin_index3_JNT':'L_Skin_index2_JNT', 'L_Skin_index2_JNT':'L_Skin_index1_JNT', 'L_Skin_index1_JNT':'L_Skin_indexPalm_JNT','L_Skin_indexPalm_JNT':'Crw_LeftHand',
                    'L_Skin_middle3_JNT':'L_Skin_middle2_JNT', 'L_Skin_middle2_JNT':'L_Skin_middle1_JNT', 'L_Skin_middle1_JNT':'L_Skin_middlePalm_JNT','L_Skin_middlePalm_JNT':'Crw_LeftHand',
                    'L_Skin_ring3_JNT':'L_Skin_ring2_JNT', 'L_Skin_ring2_JNT':'L_Skin_ring1_JNT', 'L_Skin_ring1_JNT':'L_Skin_ringPalm_JNT','L_Skin_ringPalm_JNT':'Crw_LeftHand',
                    'L_Skin_pinky3_JNT':'L_Skin_pinky2_JNT', 'L_Skin_pinky2_JNT':'L_Skin_pinky1_JNT','L_Skin_pinky1_JNT':'L_Skin_pinkyPalm_JNT','L_Skin_pinkyPalm_JNT':'Crw_LeftHand',


                    #R finger joint hierachy
                    'R_Skin_thumb3_JNT':'R_Skin_thumb2_JNT', 'R_Skin_thumb2_JNT':'R_Skin_thumb1_JNT', 'R_Skin_thumb1_JNT':'Crw_RightHand', 
                    'R_Skin_index3_JNT':'R_Skin_index2_JNT', 'R_Skin_index2_JNT':'R_Skin_index1_JNT', 'R_Skin_index1_JNT':'R_Skin_indexPalm_JNT', 'R_Skin_indexPalm_JNT':'Crw_RightHand',
                    'R_Skin_middle3_JNT':'R_Skin_middle2_JNT', 'R_Skin_middle2_JNT':'R_Skin_middle1_JNT', 'R_Skin_middle1_JNT':'R_Skin_middlePalm_JNT','R_Skin_middlePalm_JNT':'Crw_RightHand',
                    'R_Skin_ring3_JNT':'R_Skin_ring2_JNT', 'R_Skin_ring2_JNT':'R_Skin_ring1_JNT', 'R_Skin_ring1_JNT':'R_Skin_ringPalm_JNT','R_Skin_ringPalm_JNT':'Crw_RightHand',
                    'R_Skin_pinky3_JNT':'R_Skin_pinky2_JNT', 'R_Skin_pinky2_JNT':'R_Skin_pinky1_JNT','R_Skin_pinky1_JNT':'R_Skin_pinkyPalm_JNT','R_Skin_pinkyPalm_JNT':'Crw_RightHand'

                    }




        self.SkinBodyJoint_list = ['C_Skin_hip_JNT', 'C_Skin_spine1_JNT', 'C_Skin_spine2_JNT', 'C_Skin_spine3_JNT','C_Skin_chest_JNT', 'C_Skin_neck_JNT', 'C_Skin_head_JNT',]
        self.crd_BodyJoint_list = ['Crw_Hips',       'Crw_Spine',       'Crw_Spine1',       'Crw_Spine2',        'Crw_Spine3',       'Crw_Neck',        'Crw_Head',]

        self.crd_L_ArmJoint_list = [    'Crw_LeftShoulder',    'Crw_LeftArm',     'Crw_LeftForeArm',    'Crw_LeftHand',    'Crw_LeftHandSub1']
        self.crd_R_ArmJoint_list = [    'Crw_RightShoulder',   'Crw_RightArm',    'Crw_RightForeArm',   'Crw_RightHand',   'Crw_RightHandSub1']

        self.L_SkinLegJoint_list = ['L_Skin_leg_JNT', 'L_Skin_lowLeg_JNT',  'L_Skin_foot_JNT', 'L_Skin_ball_JNT',]
        self.crd_L_LegJoint_list = ['Crw_LeftUpLeg',     'Crw_LeftLeg',     'Crw_LeftFoot',   'Crw_LeftToeBase']

        self.R_SkinLegJoint_list = ['R_Skin_leg_JNT', 'R_Skin_lowLeg_JNT',  'R_Skin_foot_JNT', 'R_Skin_ball_JNT',]
        self.crd_R_LegJoint_list = ['Crw_RightUpLeg',     'Crw_RightLeg',   'Crw_RightFoot',  'Crw_RightToeBase']

        self.crd_Joint_list_All = ['Crw_Hips', 'Crw_LeftUpLeg', 'Crw_LeftLeg', 'Crw_LeftFoot', 'Crw_LeftToeBase', 'Crw_RightUpLeg', 'Crw_RightLeg', 'Crw_RightFoot', 'Crw_RightToeBase', 'Crw_Spine', 'Crw_Spine1', 'Crw_Spine2', 'Crw_Spine3', 'Crw_Neck', 'Crw_Head', 'Crw_Hair', 'Crw_LeftShoulder', 'Crw_LeftArm', 'Crw_LeftForeArm', 'Crw_LeftHand', 'Crw_LeftHandSub1', 'Crw_RightShoulder', 'Crw_RightArm', 'Crw_RightForeArm', 'Crw_RightHand', 'Crw_RightHandSub1',]
        self.HIK_Joint_list_All = [ 'Hips',     'LeftUpLeg',     'LeftLeg',    'LeftFoot',       'LeftToeBase',      'RightUpLeg',    'RightLeg',     'RightFoot',     'RightToeBase',      'Spine',    'Spine1',     'Spine2',     'Spine3',     'Neck',     'Head',     'Hair',      'LeftShoulder',    'LeftArm',     'LeftForeArm',    'LeftHand',     'LeftFingerBase',   'RightShoulder',      'RightArm',     'RightForeArm',    'RightHand',     'RightFingerBase', ]     



        ################################################################################################################################################################################################################################################################


        self.skinJNT_hierachy = {
                    #body joint hierachy
                    'C_Skin_hip_JNT':'','C_Skin_spine1_JNT':'C_Skin_hip_JNT','C_Skin_spine2_JNT':'C_Skin_spine1_JNT','C_Skin_spine3_JNT':'C_Skin_spine2_JNT',
                    'C_Skin_chest_JNT': 'C_Skin_spine3_JNT', 'C_Skin_neck_JNT':'C_Skin_chest_JNT', 'C_Skin_neckTwist_JNT':'C_Skin_neck_JNT', 'C_Skin_neckTwist1_JNT':'C_Skin_neckTwist_JNT',
                    'C_Skin_neckTwist2_JNT':'C_Skin_neckTwist1_JNT','C_Skin_head_JNT':'C_Skin_neckTwist2_JNT',

                    #R_arm joint hierachy
                    'R_Skin_shoulder_JNT':'C_Skin_chest_JNT','R_Skin_upArm_JNT':'R_Skin_shoulder_JNT',
                    'R_Skin_upArmTwist_JNT':'R_Skin_upArm_JNT','R_Skin_upArmTwist1_JNT':'R_Skin_upArmTwist_JNT','R_Skin_upArmTwist2_JNT':'R_Skin_upArmTwist1_JNT', 
                    'R_Skin_upArmTwist3_JNT':'R_Skin_upArmTwist2_JNT','R_Skin_upArmTwist4_JNT':'R_Skin_upArmTwist3_JNT', 'R_Skin_foreArm_JNT':'R_Skin_upArmTwist4_JNT',
                    'R_Skin_foreArmTwist_JNT':'R_Skin_foreArm_JNT', 'R_Skin_foreArmTwist1_JNT':'R_Skin_foreArmTwist_JNT','R_Skin_foreArmTwist2_JNT':'R_Skin_foreArmTwist1_JNT',
                    'R_Skin_foreArmTwist3_JNT':'R_Skin_foreArmTwist2_JNT', 'R_Skin_foreArmTwist4_JNT':'R_Skin_foreArmTwist3_JNT', 'R_Skin_hand_JNT':'R_Skin_foreArmTwist4_JNT',
                    'R_Skin_indexPalm_JNT':'R_Skin_hand_JNT', 'R_Skin_index1_JNT':'R_Skin_indexPalm_JNT', 'R_Skin_index2_JNT':'R_Skin_index1_JNT','R_Skin_index3_JNT':'R_Skin_index2_JNT',
                    'R_Skin_index4_JNT':'R_Skin_index3_JNT','R_Skin_middlePalm_JNT':'R_Skin_hand_JNT','R_Skin_middle1_JNT':'R_Skin_middlePalm_JNT','R_Skin_middle2_JNT':'R_Skin_middle1_JNT',
                    'R_Skin_middle3_JNT':'R_Skin_middle2_JNT','R_Skin_middle4_JNT':'R_Skin_middle3_JNT', 'R_Skin_ringPalm_JNT':'R_Skin_hand_JNT','R_Skin_ring1_JNT':'R_Skin_ringPalm_JNT',
                    'R_Skin_ring2_JNT':'R_Skin_ring1_JNT','R_Skin_ring3_JNT':'R_Skin_ring2_JNT','R_Skin_ring4_JNT':'R_Skin_ring3_JNT', 'R_Skin_pinkyPalm_JNT':'R_Skin_hand_JNT',
                    'R_Skin_pinky1_JNT':'R_Skin_pinkyPalm_JNT', 'R_Skin_pinky2_JNT':'R_Skin_pinky1_JNT', 'R_Skin_pinky3_JNT':'R_Skin_pinky2_JNT', 'R_Skin_pinky4_JNT':'R_Skin_pinky3_JNT',
                    'R_Skin_thumb1_JNT':'R_Skin_hand_JNT', 'R_Skin_thumb2_JNT':'R_Skin_thumb1_JNT', 'R_Skin_thumb3_JNT':'R_Skin_thumb2_JNT', 'R_Skin_thumb4_JNT':'R_Skin_thumb3_JNT',
                    
                    #L_arm joint hierachy
                    'L_Skin_shoulder_JNT':'C_Skin_chest_JNT','L_Skin_upArm_JNT':'L_Skin_shoulder_JNT',
                    'L_Skin_upArmTwist_JNT':'L_Skin_upArm_JNT','L_Skin_upArmTwist1_JNT':'L_Skin_upArmTwist_JNT','L_Skin_upArmTwist2_JNT':'L_Skin_upArmTwist1_JNT', 
                    'L_Skin_upArmTwist3_JNT':'L_Skin_upArmTwist2_JNT','L_Skin_upArmTwist4_JNT':'L_Skin_upArmTwist3_JNT', 'L_Skin_foreArm_JNT':'L_Skin_upArmTwist4_JNT',
                    'L_Skin_foreArmTwist_JNT':'L_Skin_foreArm_JNT', 'L_Skin_foreArmTwist1_JNT':'L_Skin_foreArmTwist_JNT','L_Skin_foreArmTwist2_JNT':'L_Skin_foreArmTwist1_JNT',
                    'L_Skin_foreArmTwist3_JNT':'L_Skin_foreArmTwist2_JNT', 'L_Skin_foreArmTwist4_JNT':'L_Skin_foreArmTwist3_JNT', 'L_Skin_hand_JNT':'L_Skin_foreArmTwist4_JNT',
                    'L_Skin_indexPalm_JNT':'L_Skin_hand_JNT', 'L_Skin_index1_JNT':'L_Skin_indexPalm_JNT', 'L_Skin_index2_JNT':'L_Skin_index1_JNT','L_Skin_index3_JNT':'L_Skin_index2_JNT',
                    'L_Skin_index4_JNT':'L_Skin_index3_JNT','L_Skin_middlePalm_JNT':'L_Skin_hand_JNT','L_Skin_middle1_JNT':'L_Skin_middlePalm_JNT','L_Skin_middle2_JNT':'L_Skin_middle1_JNT',
                    'L_Skin_middle3_JNT':'L_Skin_middle2_JNT','L_Skin_middle4_JNT':'L_Skin_middle3_JNT', 'L_Skin_ringPalm_JNT':'L_Skin_hand_JNT','L_Skin_ring1_JNT':'L_Skin_ringPalm_JNT',
                    'L_Skin_ring2_JNT':'L_Skin_ring1_JNT','L_Skin_ring3_JNT':'L_Skin_ring2_JNT','L_Skin_ring4_JNT':'L_Skin_ring3_JNT', 'L_Skin_pinkyPalm_JNT':'L_Skin_hand_JNT',
                    'L_Skin_pinky1_JNT':'L_Skin_pinkyPalm_JNT', 'L_Skin_pinky2_JNT':'L_Skin_pinky1_JNT', 'L_Skin_pinky3_JNT':'L_Skin_pinky2_JNT', 'L_Skin_pinky4_JNT':'L_Skin_pinky3_JNT',
                    'L_Skin_thumb1_JNT':'L_Skin_hand_JNT', 'L_Skin_thumb2_JNT':'L_Skin_thumb1_JNT', 'L_Skin_thumb3_JNT':'L_Skin_thumb2_JNT', 'L_Skin_thumb4_JNT':'L_Skin_thumb3_JNT',


                    #R_leg joint hierachy                    
                    'R_Skin_leg_JNT':'C_Skin_hip_JNT', 'R_Skin_legTwist_JNT':'R_Skin_leg_JNT', 'R_Skin_legTwist1_JNT':'R_Skin_legTwist_JNT', 'R_Skin_legTwist2_JNT':'R_Skin_legTwist1_JNT',
                    'R_Skin_legTwist3_JNT':'R_Skin_legTwist2_JNT', 'R_Skin_legTwist4_JNT':'R_Skin_legTwist3_JNT', 'R_Skin_lowLeg_JNT':'R_Skin_legTwist4_JNT', 'R_Skin_lowLegTwist_JNT':'R_Skin_lowLeg_JNT',
                    'R_Skin_lowLegTwist1_JNT':'R_Skin_lowLegTwist_JNT', 'R_Skin_lowLegTwist2_JNT':'R_Skin_lowLegTwist1_JNT','R_Skin_lowLegTwist3_JNT':'R_Skin_lowLegTwist2_JNT',
                    'R_Skin_lowLegTwist4_JNT':'R_Skin_lowLegTwist3_JNT', 'R_Skin_foot_JNT':'R_Skin_lowLegTwist4_JNT', 'R_Skin_ball_JNT':'R_Skin_foot_JNT','R_Skin_toe_JNT':'R_Skin_ball_JNT',

                    #R_leg joint hierachy                    
                    'L_Skin_leg_JNT':'C_Skin_hip_JNT', 'L_Skin_legTwist_JNT':'L_Skin_leg_JNT', 'L_Skin_legTwist1_JNT':'L_Skin_legTwist_JNT', 'L_Skin_legTwist2_JNT':'L_Skin_legTwist1_JNT',
                    'L_Skin_legTwist3_JNT':'L_Skin_legTwist2_JNT', 'L_Skin_legTwist4_JNT':'L_Skin_legTwist3_JNT', 'L_Skin_lowLeg_JNT':'L_Skin_legTwist4_JNT', 'L_Skin_lowLegTwist_JNT':'L_Skin_lowLeg_JNT',
                    'L_Skin_lowLegTwist1_JNT':'L_Skin_lowLegTwist_JNT', 'L_Skin_lowLegTwist2_JNT':'L_Skin_lowLegTwist1_JNT','L_Skin_lowLegTwist3_JNT':'L_Skin_lowLegTwist2_JNT',
                    'L_Skin_lowLegTwist4_JNT':'L_Skin_lowLegTwist3_JNT', 'L_Skin_foot_JNT':'L_Skin_lowLegTwist4_JNT', 'L_Skin_ball_JNT':'L_Skin_foot_JNT','L_Skin_toe_JNT':'L_Skin_ball_JNT',

                    }

        
        self.delRIG_list = ['templateJoint_GRP', 'IKJoint_GRP', 'FKJoint_GRP', 'BlendJoint_GRP', 'attach_GRP', 'space_GRP', 'stretchy_GRP', 'twist_GRP', 'roll_GRP', 'noneFlip_GRP', 'place_NUL','control_GRP', 'auxillary_GRP','psd_GRP']

        self.crdBaseJnt_hight = 8.923

        self.nameSpace = self.getNamespace()


        ####### initial pose ############################################################################################################################################################################################################################

        self.initPoseCtrl_list = ['L_IK_handVec_CON', 'R_IK_handSub_CON', 'R_IK_hand_CON', 'R_IK_handVec_CON', 'C_IK_upBodyRot1_CON', 'C_IK_upBodyRot2_CON', 'R_FK_pinky2_CON', 'R_FK_pinky1_CON', 'R_FK_ring3_CON', 'R_FK_ring2_CON', 'R_FK_ring1_CON', 'R_FK_middle3_CON', 'R_FK_middle2_CON', 'R_FK_thumb1_CON', 'L_FK_pinky3_CON', 'L_FK_pinky2_CON', 'L_FK_pinky1_CON', 'L_FK_ring3_CON', 'L_FK_ring2_CON', 'L_FK_ring1_CON', 'R_legBlend_CON', 'L_legBlend_CON', 'R_armBlend_CON', 'L_armBlend_CON', 'R_FK_middle1_CON', 'R_FK_index3_CON', 'R_FK_index2_CON', 'R_FK_index1_CON', 'R_FK_thumb3_CON', 'R_FK_thumb2_CON', 'R_FK_pinky3_CON', 'L_FK_thumb3_CON', 'L_FK_thumb2_CON', 'L_FK_thumb1_CON', 'R_FK_shoulder_CON', 'L_FK_shoulder_CON', 'R_IK_footVec_CON', 'R_IK_footRollUp_CON', 'R_IK_ballRollUp_CON', 'L_FK_middle3_CON', 'L_FK_middle2_CON', 'L_FK_middle1_CON', 'L_FK_index3_CON', 'L_FK_index2_CON', 'L_FK_index1_CON', 'L_IK_footSub_CON', 'L_IK_foot_CON', 'C_IK_neck_CON', 'C_IK_head_CON', 'L_IK_hand_CON', 'L_IK_handSub_CON', 'R_IK_ball_CON', 'R_IK_footSub_CON', 'R_IK_foot_CON', 'L_IK_footVec_CON', 'L_IK_footRollUp_CON', 'L_IK_ballRollUp_CON', 'L_IK_ball_CON', 'move_CON', 'root_CON', 'direction_CON', 'place_CON', 'C_IK_lowBody_CON', 'C_IK_upBody_CON', 'C_IK_hip_CON']

        self.initPoseAtt_list = { 'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0, 'sx': 1, 'sy': 1, 'sz': 1,'v': 1 }
                                 
        #################################################################################################################################################################################################################################################




        ####### constraint contrioler list ############################################################################################################################################################################################################################

        self.const_list = [('Hips','root_CON'),('Spine3','C_IK_upBody_CON'),('Neck','C_IK_neck_CON'),('Head','C_IK_head_CON'),('Spine1', 'C_IK_upBodyRot1_CON'),('Spine2','C_IK_upBodyRot2_CON'),('Spine','C_IK_lowBody_CON'),('Hips','C_IK_hip_CON'),('LeftShoulder','L_FK_shoulder_CON'),('RightShoulder','R_FK_shoulder_CON'),('LeftHand','L_IK_hand_CON'),('RightHand','R_IK_hand_CON'),('LeftToeBase','L_IK_ball_CON'),('LeftFoot','L_IK_foot_CON'),('RightToeBase','R_IK_ball_CON'),('RightFoot','R_IK_foot_CON')]
        self.pole_list = [('LeftForeArm','L_IK_handVec_CON'),('RightForeArm','R_IK_handVec_CON'),('LeftLeg','L_IK_footVec_CON'),('RightLeg','R_IK_footVec_CON')]






    def getNamespace(self):
        name_space = []
        listCtrls = cmds.ls(sl=1)

        if(listCtrls == []):
            cmds.error("you must select the controler or the object")
        else:
            sign =  listCtrls[0].find(':')
            if(sign == -1):
                return ''

            else:
                for ctrl in listCtrls:
                    name_space.append(ctrl.split(":")[0])

                return name_space[0]



