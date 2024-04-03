#coding:utf-8
'''
@ author    : daeseok.chae
@ date      : 2018.11.22
@ comment   : BVH파일을 만들어주는 모듈입니다.
'''
import maya.cmds as cmds
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import os
import math

# fps별 name by fps 테이블
TIME_MAP = {'game':15.0, 'film':24.0, 'pal':25.0, 'ntsc':30.0, 'show':48.0, 'palf':50.0, 'ntscf':60.0}
                
class BVHExporter():
    '''
    Export BVH file module.
    :param
    '''
    def __init__(self):
        self.remap = False
        self.remapData = {}
        self.joints = []

    def print_blocks_JSON( self, jointname, indent):
        print_string = ""
        self.joints.append(jointname)

        tr = cmds.xform(jointname, q=True, t=True)

        if self.remap:
            if jointname.split("|")[-1].split(":")[-1] in self.remapData:
                name = self.remapData[jointname.split("|")[-1].split(":")[-1]]
        else:
            name = jointname.split("|")[-1].split(":")[-1]

        if cmds.listRelatives( jointname, p=1, type='joint' ) == None:
            print_string = ""
            print_string += indent + "ROOT %s" % name + "\n"
        else:
            print_string += indent + "JOINT %s" %name + "\n"

        print_string += indent + "{" + "\n"
        print_string += indent + "  " + "OFFSET "
        print_string += str(tr[0]) + " " + str(tr[1]) + " " + str(tr[2]) + "\n"
        print_string += indent + "  " + "CHANNELS 6 "
        print_string += "Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"

        return print_string

    def write_key( self, writer, start, end):
        f = start
        while f <= end:
            for j in self.joints: 
                selection = om.MSelectionList()
                selection.add( j )
                dagPath = om.MDagPath()
                selection.getDagPath(0, dagPath)
                mobj = om.MObject()
                selection.getDependNode(0, mobj)
                transFn = om.MFnTransform(dagPath)
                
                # localMatrix = worldMatrix * parentInverseMatrix

                mtxAttr = transFn.attribute( 'worldMatrix' )
                mtxPlug = om.MPlug( mobj, mtxAttr )
                mtxPlug = mtxPlug.elementByLogicalIndex( 0 )
                frameCtx = om.MDGContext( om.MTime( f, om.MTime.uiUnit()) )       
                mtxObj   = mtxPlug.asMObject( frameCtx )
                mtxData  = om.MFnMatrixData( mtxObj )
                mtxValue = mtxData.matrix()

                wrdMtx = mtxValue

                # get parent inverse matrix
                mtxAttr = transFn.attribute('parentInverseMatrix')
                mtxPlug = om.MPlug( mobj, mtxAttr )
                mtxPlug = mtxPlug.elementByLogicalIndex( 0 )
                frameCtx = om.MDGContext( om.MTime( f, om.MTime.uiUnit()) )       
                mtxObj   = mtxPlug.asMObject( frameCtx )
                mtxData  = om.MFnMatrixData( mtxObj )
                mtxValue = mtxData.matrix()

                piMtx = mtxValue

                localMtx = wrdMtx * piMtx
                
                tmtx = om.MTransformationMatrix( localMtx )
                tr = tmtx.translation(om.MSpace.kWorld)
                rot = tmtx.rotation().asEulerRotation()
                x,y,z = self.getAngle(rot[0], rot[1], rot[2])
                                      
                print_string = str(tr[0]) + " " + str(tr[1]) + " " + str(tr[2]) + " "
                print_string += str(z) + " " + str(x) + " " + str(y) + " "
                writer.write(print_string)
                
            writer.write("\n")
            f += 1          
            
    def getAngle( self, x, y, z):
        if ( x <= 0.0 and x > -0.000000000001 ):  x = 0.0
        if ( y <= 0.0 and y > -0.000000000001 ):  y = 0.0
        if ( z <= 0.0 and z > -0.000000000001 ):  z = 0.0
        
        A = math.cos(x)
        B = math.sin(x)
        C = math.cos(y)
        D = math.sin(y)
        E = math.cos(z)
        F = math.sin(z)

        x = math.degrees( math.asin( B*C ))
        y = math.degrees(math.atan2( D, A*C ))
        z = math.degrees(math.atan2( -1*B*D*E + A*F, B*D*F + A*E ))

        x = self.noScience(x)
        y = self.noScience(y)
        z = self.noScience(z)

        return x,y,z

    def noScience( self, num ):
        if ( num > -0.1 and num < -1.0 ):
            num = num/100000
        else:
            num = num*100000

        tnum = round(num)
        if ( ( num - tnum ) >= 0.5 ):
            num = math.ceil(num)
        else:
            num = math.floor(num)

        num = num/100000
        return num

    def recursive_write_joint(self, writer, jointname, indent, last=False):
        '''
        "*_root_JNT"부터 hierachy를 쭉 타고 내려가면서  Joint 정보를 작성합니다.
        :param writer: 작성중인 파일의 파일 포인터
        :param jointname: 현재 조인트 이름
        :param indent: 인터벌이 얼마나 필요한지
        :param last: 끝인가?
        :return:
        '''
        #writer.write(indent+"{\n")
        if self.remap:
            if jointname.split("|")[-1].split(":")[-1] in self.remapData:
                pass
            else:
                return

        writer.write( self.print_blocks_JSON(jointname, indent))

        children = cmds.listRelatives(jointname, c=1, type='joint', f=1)

        if children:
            num = 0
            for c in children:
                if num + 1 == len(children):
                    self.recursive_write_joint(writer, c, indent + "  ", True)
                else :
                    self.recursive_write_joint(writer, c, indent + "  ", False)
                num += 1
        else:
            print_string = ""
            print_string += indent+"  End Site\n"
            print_string += indent+"  {\n"
            print_string += indent+"    OFFSET 0 0 0\n"
            print_string += indent+"  }\n"
            writer.write(print_string)

        if not last:
            writer.write(indent + "}\n")
        if last:
            writer.write(indent + "}\n")


    def generate_joint_data( self, rootJoint="", filepath="", start=0, end=0 ):
        '''
        Joint정보를 bvh파일로 작성합니다.
        :param selected: root_JNT 노드.
        :param filepath: bvh 파일 경로
        :param start: export하고자 하는 데이터의 startFrame
        :param end: export하고자 하는 데이터의 endFrame
        :return: X
        '''
        if not rootJoint:
            rootJoint = cmds.ls(sl=True)[0]

        if not filepath:
            print 'export failure'
            return
            
        if not os.path.exists( os.path.dirname(filepath) ):
            os.makedirs(os.path.dirname(filepath))
        writer = open(filepath, 'w')
        if os.path.splitext(filepath)[1] == ".bvh":
            if not start:
                start = cmds.playbackOptions(q=True, min=True)
            if not end:
                end = cmds.playbackOptions(q=True, max=True)
            timeUnit = cmds.currentUnit( t=True, q=True )
            frameTime = 1/float(TIME_MAP[timeUnit])

            writer.write( "HIERARCHY\n" )
            self.recursive_write_joint(writer, rootJoint, "", True)
            writer.write( "MOTION\n" )
            writer.write( "Frames: %s\n" % str(end-start+1) )
            writer.write( "Frame Time: %s\n" % frameTime )
            r = None
            try:
                attrs = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
                r = cmds.bakeSimulation(self.joints, t=(start, end), at=attrs)
            except:
                pass

            self.write_key(writer, start, end)
            if r:
                cmds.undo()

        writer.close()
        while len(self.joints) > 0 : self.joints.pop()
