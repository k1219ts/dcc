# encoding=utf-8
import os
import json
import time
import logging
import shutil
import string
import subprocess
import maya.cmds as cmds
import aniCommon
from PySide2 import QtWidgets, QtCore, QtGui


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def hconv(text):
    return unicode(text, 'utf-8')


class MakeMovie():
    def __init__(self, options):
        self.options = options
        self.getDxConfigPath()


    def getDxConfigPath(self):
        if os.environ.has_key('DXCONFIGPATH'):
            configFile = os.path.join(os.environ['DXCONFIGPATH'], 'Project.config')

            if os.path.isfile(configFile):
                print 'DXCONFIGPATH:', configFile

                with open(configFile, 'r') as f:
                    configData = json.load(f)
                if configData.has_key('mayaMOV'):
                    self.options['fps'] = configData['mayaMOV']['fps']
                    self.options['codec'] = configData['mayaMOV']['codec']
                    return

        playbackOptions = {'game': 15, 'film': 24, 'pal': 25, 'ntsc': 30,
                           'show': 48, 'palf': 50, 'ntscf': 60}

        unit = cmds.currentUnit(query=True, time=True)
        if playbackOptions.has_key(unit):
            self.options['fps'] = playbackOptions[unit]
        else:
            unit = unit.replace('fps', '').replace(' ', '')
            self.options['fps'] = unit
        self.options['codec'] = 'h264'


    def getPathdata(self, outpath):
        movieDir = os.path.dirname(outpath)
        prefix = os.path.splitext(os.path.basename(outpath))[0]
        t = int(time.mktime(time.gmtime()))
        tempDir = os.path.join(movieDir, prefix + "_" + str(t))
        fileName = os.path.join(tempDir, prefix)

        if not os.path.isdir(tempDir):
            os.makedirs(tempDir)
        return {'filename': fileName, 'tempDir': tempDir, 'prefix': prefix}


    def playblast(self,
                  parentWidget,
                  fromWindow,
                  type='movie',
                  addstamp=False):
        """

        :param parentWidget: parentWidget Qt Widget
        :type fromWindow: bool
        :type options: dict
        :param saveFile: movie file name
        :param type: If 'movie', convert sequence to movie file
        :param addstamp: If True, Add nuke stamp
        """
        viewer_state = True
        playblast_kwargs = dict()
        pathData = dict()

        if self.options['outpath']:
            msg = QtWidgets.QMessageBox.question(
                parentWidget,
                hconv("실행안내"),
                hconv("{0} 파일을 만드시겠습니까?".format(type)),
                QtWidgets.QMessageBox.Ok,
                QtWidgets.QMessageBox.Cancel)

            if not msg == QtWidgets.QMessageBox.Ok: return
            # fileName, tempDir, prefix = self.getPathdata(self.options['outpath'])
            pathData = self.getPathdata(self.options['outpath'])
            viewer_state = False

        if self.options['seqTime']:
            logger.debug(u'[Playblast] Camera sequencer mode')
            self.options['startTime'] = self.options['seqStartTime']
            self.options['endTime'] = self.options['seqEndTime']
            playblast_kwargs['sequenceTime'] = True

        playblast_kwargs['filename'] = pathData.get('filename', None)
        playblast_kwargs['startTime'] = self.options['startTime']
        playblast_kwargs['endTime'] = self.options['endTime']
        playblast_kwargs['viewer'] = viewer_state
        playblast_kwargs['offScreen'] = self.options['offScreen']

        if not fromWindow:
            playblast_kwargs['widthHeight'] = [self.options['width'], self.options['height']]

        # if self.options.has_key('sound'):
        #     self.movie(tempDir = pathData["tempDir"], **playblast_kwargs)
        # else:
        captureFileName = self.capture(**playblast_kwargs)

        if self.options['outpath'] and type == 'movie' and not addstamp:
            self.convert(tempDir=pathData['tempDir'], prefix=pathData['prefix'])
        # elif self.options['outpath'] and type == 'movie' and addstamp:
        #     logger.debug(u'[Add Stamp] Send job to tractor')
        #     showName = aniCommon.getShowShot(self.options['mayafile'])[0]
        #     soundFile = ""
        #     # if self.options.has_key("sound"):
        #     #     soundFile = self.options['sound']
        #     job = tractorNuke.JobScript(
        #         platePath=pathData['tempDir'],
        #         m_shotName=self.options['scene'],
        #         m_project=showName,
        #         userName=self.options['artist'],
        #         startFrame=self.options['startTime'],
        #         endFrame=self.options['endTime'],
        #         isRetime=self.options['compRetime'],
        #         sound=soundFile
        #     )
        #     job.spool()

    # def movie(self, tempDir, **kwargs):
    #     logger.debug(u'[Playblast] Sequence : {0}\n'.format(tempDir))
    #     logger.debug(u'[Playblast] Options : {0}\n'.format(self.options))
    #     kwargs['filename'] = os.path.join(os.path.dirname(tempDir), os.path.basename(self.options['outpath']))#self.options['outpath']
    #     captureFileName = cmds.playblast(
    #         format="movie",
    #         width = self.options['width'] if self.options['width'] % 2 == 0 else self.options['width'] - 1,
    #         height = self.options['height'] if self.options['height'] % 2 == 0 else self.options['height'] - 1,
    #         forceOverwrite=True,
    #         clearCache=True,
    #         showOrnaments=True,
    #         percent=100,
    #         s = self.options['sound'],
    #         compression="jpg",
    #         **kwargs
    #     )
    #     # print captureFileName, tempDir
    #
    #     # cmd = '/backstage/bin/DCC rez-env ffmpeg-4.2.0 -- ffmpeg'
    #     # cmd += ' -i %s' % captureFileName
    #     # cmd += ' -y %s' % self.options['outpath']
    #     # p = subprocess.Popen(cmd, env=self.virtualenv, shell=True)
    #     # p.wait()
    #
    #     if tempDir:
    #         os.system("rm -rf %s" % tempDir)
    #
    #     return captureFileName

    def capture(self, **kwargs):
        logger.debug(u'[Playblast] Sequence : {0}\n'.format(kwargs['filename']))
        logger.debug(u'[Playblast] Options : {0}\n'.format(self.options))
        captureFileName = cmds.playblast(
            format="image",
            forceOverwrite=True,
            clearCache=True,
            showOrnaments=True,
            framePadding=4,
            percent=100,
            compression="jpg",
            **kwargs
        )
        return captureFileName


    def convert(self, tempDir, prefix):
        """Convert Sequence Files To Movie File, Using ffmpeg

        :param options: Options From Maya Scene
        :param tempDir: Temp path of playblast image sequences
        :param prefix: Prefix for movie file name
        """
        logger.debug(u'[Convert] Source : {0}'.format(tempDir))
        logger.debug(u'[Convert] Output : {0}\n'.format(self.options['outpath']))
        cmd = '%s rez-env python-2 ffmpeg_toolkit -- ffmpeg_converter' % os.environ['DCCPROC']

        cmd += ' -c {codec}'
        cmd += ' -r {fps}'
        cmd += ' -i {blastTempDir}/{prefix}.%04d.jpg'
        cmd += ' -o {fileName}'
        cmd += ' -s {width}x{height}'
        cmd = cmd.format(
            codec=self.options['codec'],
            fps=self.options['fps'],
            blastTempDir=tempDir,
            prefix=prefix,
            width=self.options['width'] if self.options['width'] % 2 == 0 else self.options['width'] - 1,
            height=self.options['height'] if self.options['height'] % 2 == 0 else self.options['height'] - 1,
            fileName=self.options['outpath']
        )

        # add audio
        audio = cmds.ls(type='audio')
        if audio:
            audioFile = cmds.getAttr(audio[0] + '.filename')
            audioOffset = int(cmds.getAttr(audio[0] + '.offset'))
            logger.debug('[find Audio] : %s %s' % (audioFile, audioOffset))

            cmd += ' -au -a {audioFile} -af {audioOffset}'.format(audioFile=audioFile,
                                                                  audioOffset=audioOffset)

        p = subprocess.Popen(cmd, shell=True, env=dict())
        p.wait()
        logger.debug(u'[Convert] Command : {0}'.format(cmd))
        if self.options['remove_sequence']:
            shutil.rmtree(tempDir)
            logger.debug(u'[Convert] Temp folder {0} removed'.format(tempDir))

        # 다른 작업창에서 작업중일때 messagebox와 마야가 함께 따라오는 문제가있어서
        # parent widget 을 None으로 변경.
        QtWidgets.QMessageBox.information(None, hconv("알림"),
                                          hconv("성공적으로 완료되었습니다."))
        logger.debug(u'[Convert] Finished')
