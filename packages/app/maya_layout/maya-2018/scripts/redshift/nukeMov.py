#encoding=utf-8
#!/usr/bin/env python
import os
import sys
import nuke, nukescripts
import subprocess
import stat
import shutil

class NukeMov():
    def __init__(self):
        self.platePath   = sys.argv[1] # /asset/redshift/ani/images/ABC_0010_ani_v01_w01/shot/filename = camera
        self.movOutPath  = sys.argv[2] # /asset/redshift/ani/preview/DEEkey_0010_ani_v01_w02_test_layer_shot1_persp_rs.mov
        self.startFrame  = sys.argv[3]
        self.endFrame    = sys.argv[4]
        self.shotName    = sys.argv[5] # ABC_0010_ani_v01_w01 ---> maya scene file name
        self.project     = sys.argv[6]
        self.USERNAME    = sys.argv[7]
        self.isRetime    = sys.argv[8]
        self.fps         = sys.argv[9]
        self.cameraName  = sys.argv[10]
        self.focalLength = sys.argv[11]
        self.metadataStr = sys.argv[12]
        self.out_format  = sys.argv[13] # exr
        self.imageplane  = sys.argv[14]
        self.useCamSeq   = sys.argv[15]
        self.movFormat = sys.argv[17] # A or B
        self.cameraType = sys.argv[18]
        self.speedck = sys.argv[19]

        self.runMovMake()

    def runMovMake(self):
        self.outFilePath()
        self.shotNameSet()
        self.gizmoSet()
        self.outputMov()
        self.nukeSet()

    def outFilePath(self):
        # plateDir : /asset/redshift/ani/images/ABC_0010_ani_v01_w01/shot/
        # plateFile : camera
        plateDir    = os.path.dirname(self.platePath)
        plateFile   = os.path.basename(self.platePath)
        tmpBA_prefix = 'tmpBASE'
        self.tmpBA     = plateDir + "/%s/%s.####.%s" % (tmpBA_prefix, plateFile, self.out_format)
        self.tmpBA_chk = self.tmpBA.replace('####', str(int(self.startFrame)).zfill(4))

        tmpFG_prefix = 'tmpFG'
        self.tmpFG     = plateDir + "/%s/%s.####.%s" % (tmpFG_prefix, plateFile, self.out_format)
        self.tmpFG_chk = self.tmpFG.replace('####', str(int(self.startFrame)).zfill(4))

        tmpBG_prefix = 'tmpBG'
        self.tmpBG     = plateDir + "/%s/%s.####.%s" % (tmpBG_prefix, plateFile, self.out_format)
        self.tmpBG_chk = self.tmpBG.replace('####', str(int(self.startFrame)).zfill(4))

    def shotNameSet(self):
        if self.useCamSeq == '1':
            self.shotName = sys.argv[16]
            self.shotNameS = self.shotName

        # sceneFileName = self.shotName
        self.shotNameS   = '_'.join( self.shotName.split('_')[:2] ) # ABC_0010
        # Mov File Name
        self.shotNameP   = os.path.basename(self.movOutPath)[:-4] # ABC_0010_ani_v01_w01_layer1_rs

    def gizmoSet(self):
        # Gizmo 수정은 아래 경로에 있는 파일을 수정합니다.
        # /netapp/backstage/pub/apps/nuke/Team_CMP/Gizmo/stamp_redshift.gizmo
        self.stampType = 'stamp_redshift'
        if self.cameraType == 'Fisheye' and self.speedck == 'ok':
            self.stampType = 'stamp_fisheye_redshift'
        elif self.speedck == 'ok':
            self.stampType = 'stamp_redshift_sj'

    def outputMov(self):
        # A is 1280x720, B is 1920x1080
        self.movFormatDict = {'A': 0, 'B': 1}
        self.focalLength = '{0}mm ({1}fps)'.format(self.focalLength.split('.')[0], self.fps.split('.')[0])
        # 폴더 구조 설정
        platePathSplit = self.platePath.split(os.sep)
        shotPath = os.sep.join(platePathSplit[:-4])
        self.renderScriptRoot = os.sep.join([shotPath, "preview"])

    def nukeSet(self):
        nuke.root()['first_frame'].setValue(int(self.startFrame))
        nuke.root()['last_frame'].setValue(int(self.endFrame))

        # Imageplane이 있는 경우 처리
        chkFG = int( self.imageplane.split(':')[0] )
        chkBG = int( self.imageplane.split(':')[1] )

        # 파일 로딩 에러 나는 경우 처리 방법 선택
        on_error_types = ['black', 'checkerboard', 'nearest frame']
        on_error_type = on_error_types[0] # black

        if chkFG > 0 and chkBG > 0:
            if os.path.exists(self.tmpBA_chk):
                plateNode = nuke.nodes.Read(file=self.tmpBA, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                renderNode = plateNode

            if os.path.exists(self.tmpFG_chk) and os.path.exists(self.tmpBG_chk):
                imageplaneNodeFG = nuke.nodes.Read(file=self.tmpFG, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                imageplaneNodeBG = nuke.nodes.Read(file=self.tmpBG, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                imgMergeNode = nuke.nodes.Merge(name='Merge_ImageplaneFG', inputs=[plateNode, imageplaneNodeFG])
                renderNode = nuke.nodes.Merge(name='Merge_Imageplane', inputs=[imageplaneNodeBG, imgMergeNode])

        elif chkFG > 0:
            if os.path.exists(self.tmpBA_chk):
                plateNode = nuke.nodes.Read(file=self.tmpBA, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                renderNode = plateNode

            if os.path.exists(self.tmpFG_chk):
                imageplaneNodeFG = nuke.nodes.Read(file=self.tmpFG, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                renderNode = nuke.nodes.Merge(name='Merge_Imageplane', inputs=[plateNode, imageplaneNodeFG])

        elif chkBG > 0:
            if os.path.exists(self.tmpBA_chk):
                plateNode = nuke.nodes.Read(file=self.tmpBA, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                renderNode = plateNode

            if os.path.exists(self.tmpBG_chk):
                imageplaneNodeBG = nuke.nodes.Read(file=self.tmpBG, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
                renderNode = nuke.nodes.Merge(name='Merge_Imageplane', inputs=[imageplaneNodeBG, plateNode])
        else:
            plateNode = nuke.nodes.Read(file=self.platePath + ".####.%s" % self.out_format, first=self.startFrame, last=self.endFrame, on_error=on_error_type)
            renderNode = plateNode

        self.nukeNode(renderNode)

    def nukeNode(self, renderNode = None):
        stampNode = nuke.createNode(self.stampType)
        stampNode.setInput(0, renderNode)

        stampNode['Artist_name'].setValue(self.USERNAME)
        stampNode['Shotname'].setValue(self.shotNameS) # ABC_0010
        stampNode['Camera'].setValue(self.cameraName.split(':')[0].replace('Shape', '')) # Camera1
        stampNode['FocalLength'].setValue(self.focalLength) # 24mm

        try:
            stampNode.node('P_INPUT1')['message'].setValue('')
        except:
            pass
        stampNode['Project_name'].setValue(self.project)

        stampNode['formatsize'].setValue(self.movFormatDict[self.movFormat])
        writeNode = nuke.nodes.Write(file="{0}/{1}/{2}.####.jpg".format(self.renderScriptRoot,
                                     self.shotNameP, self.shotNameP),
                                     file_type='jpeg') # preview/ABC_0010_ani_v01_w01_filename_rs/ABC_0010_ani_v01_w01_filename_rs.####.jpg
        writeNode['_jpeg_quality'].setValue(1)
        writeNode['_jpeg_sub_sampling'].setValue('4:2:2')
        writeNode.setInput(0, stampNode)

        nukescripts.clear_selection_recursive()
        renderScriptPath = '{0}/precomp/{1}.nk'.format(self.renderScriptRoot, self.shotNameP) # preview/ABC_0010_ani_v01_w01_filename_rs.nk

        if not(os.path.exists(os.path.dirname(renderScriptPath))):
            os.makedirs(os.path.dirname(renderScriptPath))
        nuke.scriptSaveAs(renderScriptPath, overwrite=1)

        writePath = writeNode['file'].value()
        writeRoot = os.path.dirname(writePath)

        if not (os.path.exists(writeRoot)):
            os.makedirs(writeRoot)

        nuke.execute(writeNode.name(),
                     int(nuke.root()['first_frame'].value()),
                     int(nuke.root()['last_frame'].value()))

        bpr = '9000k'
        ffCmd = ['/opt/ffmpeg/bin/ffmpeg', '-r', self.fps, '-start_number',
                 str(nuke.root()['first_frame'].value()), '-i', writePath ]
        ffCmd += ['-r', self.fps, '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'slow']
        ffCmd += ['-profile:v', 'baseline', '-b', bpr, '-tune', 'zerolatency']
        ffCmd += ['-metadata', 'title=%s' % self.metadataStr]
        ffCmd += ['-y', self.movOutPath]
        # HQ is 16000k-14000k, Normal is 9000k, LT is 6000k
        subprocess.call(ffCmd)
        self.cleanUP(writeRoot)

    def cleanUP(self, writeRoot = None):
        # cleanup preview .jpg
        if os.path.exists(writeRoot):
            self.remove_dir_tree(writeRoot)

    def remove_dir_tree(self, remove_dir = None):
        # 하위 폴더 삭제
        try:
            shutil.rmtree(remove_dir, ignore_errors=False, onerror=self.remove_readonly)
            print '[Debug] removed : %s' % remove_dir
        except(PermissionError) as e:
            print '[Debug] %s - %s' % (e.filename, e.strerror)

    def remove_readonly(self, func = None, path = None, excinfo = None):
        os.chmod(path, stat.S_IWRITE)
        func(path)

if __name__ == '__main__':
    nukemov = NukeMov()
