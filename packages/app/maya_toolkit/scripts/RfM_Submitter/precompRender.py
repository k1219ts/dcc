#encoding=utf-8
#!/usr/bin/env python
"""
THIS SCRIPT IS ABOUT TO EXECUTED BY "NUKE -T" COMMAND
Nuke Read node render image update for lighting and fx render
"""

import nuke
import os
import sys
import re
import glob
import pprint
import getpass
import argparse


if sys.platform == 'linux2':
    sys.path.append('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')

elif sys.platform == 'darwin':
    sys.path.append('/Volumes/10.0.0.248/backstage/pub/apps/tractor/linux/Tractor-2.0/lib/python2.7/site-packages')
import tractor.api.author as author

from ImageParser import ImageParser

def getNukeVersion(nukefile):
    fileVersion = open(nukefile).readline().split('/')[3]
    # Nuke10.0v4
    return fileVersion


def GetOutputScriptFile(inScript):
    '''
    Create last version nuke-script filename
    '''

    fn = os.path.splitext(inScript)[0]
    fn = re.sub('_v\d+', '', fn)
    fn = re.sub('_w\d+', '', fn)

    source = glob.glob(fn+'*')
    # pprint.pprint(source)

    versions = list()
    for f in source:
        if os.path.isfile(f) and os.path.splitext(f)[-1] == '.nk':
            p = re.compile(r'_v\d+').findall(f)
            if p:
                versions.append(int(p[0][2:]))
    versions.sort()

    if versions:
        new_version = versions[-1] + 1
    else:
        new_version = 1

    newScript = fn + '_v%03d' % new_version + '.nk'
    return newScript


def GetReadNodesData():
    '''
    Find read node and
        create ImageParser class data
    '''
    data = dict()
    for n in nuke.allNodes('Read'):
        cfile = n.knob('file').value()
        cfile = PrefixRemove(cfile)

        img = ImageParser()
        img.fileParser(cfile)
        data[n.name()] = img
    return data


def GetAllConnected(nodes, filter=None):
    allDeps = set()
    filteredNode = set()
    depsList = nodes
    evaluateAll = True
    while depsList:
        #deps = nuke.dependencies(depsList, _nuke.INPUTS | _nuke.HIDDEN_INPUTS)
        #deps += nuke.dependentNodes(_nuke.INPUTS | _nuke.HIDDEN_INPUTS, depsList, evaluateAll)
        deps = nuke.dependentNodes(_nuke.INPUTS | _nuke.HIDDEN_INPUTS, depsList, evaluateAll)
        evaluateAll = False
        depsList = []
        for i in deps:
            if i not in allDeps:
                depsList.append(i)
                allDeps.add(i)
            if filter:
                if i.Class() == filter:
                    filteredNode.add(i)
    if filter:
        return filteredNode
    else:
        return allDeps

def PrefixRemove(path):
    if path.startswith('/netapp/dexter/show'):
        path = path.replace('/netapp/dexter/show', '/show')
    return path

def removeDuplicatedPathNode(nodes):
    pathDic = {}
    for i in nodes:
        absw = i['file'].value()
        if pathDic.has_key(absw):
            pass
        else:
            pathDic[absw] = i

    return pathDic.values()


def sendRenderFarm(nkfile, writeNodes):
    startFrame = int(nuke.root()['first_frame'].value())
    endFrame = int(nuke.root()['last_frame'].value())
    svc = 'nuke'
    fps = '24'
    framePerTask = 4

    nukeVer = getNukeVersion(nkfile)
    nukeexec = nukeVer.split('v')[0]
    slot = 1

    job = author.Job(title='(renderBot)_'+os.path.basename(nkfile),
                     priority=1,
                     service=svc,
                     tags=['team'],
                     tier='COMP',
                     projects=['comp'],
                     envkey=['nuke']
                     )

    for writeNode in writeNodes:
        # RAW WRITE PATH
        rawPath = writeNode['file'].value()
        if rawPath.startswith('/netapp/dexter'):
            rawPath = rawPath.replace('/netapp/dexter', '')

        if not (os.path.exists(os.path.dirname(rawPath))):
            os.makedirs(os.path.dirname(rawPath))

        # STEREO CHECK
        viewList = writeNode['views'].value().split(' ')
        if len(viewList) > 1:
            isStereo = True
        else:
            isStereo = False

        if writeNode['file'].value().endswith('.jpg'):
            # JPG ITEM RENDER WITH PREVIEW MOV
            rootJob = author.Task(title='FFMPEG MOV')
            for j in viewList:
                if "%V" in rawPath:
                    jpgFilepath = rawPath.replace("%V", j)
                elif "%v" in rawPath:
                    jpgFilepath = rawPath.replace("%v", j[0])
                else:
                    jpgFilepath = rawPath
                outputPath = os.path.join(os.path.dirname(os.path.dirname(jpgFilepath)), jpgFilepath.split('/')[-1].split('.')[0]) + '.mov'

                jobArg = movSetting(fps, startFrame, jpgFilepath, outputPath)
                movCommand = author.Command(argv=jobArg)
                rootJob.addCommand(movCommand)
        else:
            rootJob = author.Task(title='DONE')

        for j in range(startFrame, endFrame + 1, framePerTask):
            firstJobFrame = j
            if j + framePerTask > endFrame:
                lastJobFrame = endFrame
            else:
                lastJobFrame = j + framePerTask - 1

            subTask = author.Task(title="%s, %s" % (firstJobFrame, lastJobFrame))
            cmd = "/usr/local/%s/%s -t -F %s,%s -X %s %s" % (nukeVer, nukeexec, firstJobFrame,
                                                             lastJobFrame, writeNode.name(),
                                                             nkfile)
            subCmd = author.Command(argv= cmd,
                                    envkey=['nuke'],
                                    service=svc)
            subTask.addCommand(subCmd)

            subTask.atleast = slot
            rootJob.addChild(subTask)

        job.addChild(rootJob)

    author.setEngineClientParam(hostname='10.0.0.106',
                                port=80,
                                user=getpass.getuser(),
                                debug=True)
    job.spool()
    author.closeEngineClient()


def movSetting(fps, startFrame, renderPath, outputPath):
    jobArg = '/opt/ffmpeg/bin/ffmpeg -r %s -start_number ' % fps
    jobArg += '%s -i %s -r %s -an -vcodec libx264 ' % ( startFrame, renderPath, fps)
    jobArg += '-pix_fmt yuv420p -preset slow -profile:v '
    jobArg += 'baseline -b 30000k -tune zerolatency '
    jobArg += '-y %s' % outputPath
    return jobArg


class ImageUpdate:
    '''
    Read node find and update main process
    '''
    def __init__(self, args):
        self.args = args
        self.debug = args.verbose
        self.args.newScript = GetOutputScriptFile(args.nukeScript)

        self.doIt()

    def doIt(self):
        self.uimg = ImageParser()
        self.uimg.versionPathParser(self.args.imagePath)

        # nuke file open
        nuke.scriptOpen(self.args.nukeScript)

        # node process - iterate read-node
        updatedNodes = self.nodeProc()
        print updatedNodes
        writeNodeList = None

        if updatedNodes:
            writeNodeList = GetAllConnected(updatedNodes, 'Write')
            writeNodeList = removeDuplicatedPathNode(writeNodeList)

            if writeNodeList:
                # WRITE NODE PATH!!
                for w in writeNodeList:
                    nukescripts.clear_selection_recursive()
                    w['selected'].setValue(True)

                    # TODO: NEED TO CHANGE WRITE NODE FILE PATH NAME???
                    nukescripts.version_up()

        # nuke file save
        nuke.scriptSave(self.args.newScript)
        print '#-------------------#'
        print ' >> New Nuke-script : ', self.args.newScript
        print '#-------------------#'
        if writeNodeList:
            print "updated Write Nodes", writeNodeList
            # SEND TRACTOR TO RENDER
            sendRenderFarm(self.args.newScript, writeNodeList)

    def updateDebugPrint(self, node, cfile, ufile):
        if not self.debug:
            return
        if not ufile:
            return
        print '  --> %s' % node
        print '\tcurrent :', cfile
        print '\tupdate  :', ufile


    def setFilename(self, node, filename):
        readNode = nuke.toNode(node)
        readNode.knob('file').setValue(filename)

    def updateProc(self, node, cfile, ufile):
        if cfile == ufile:
            return
        elif not ufile:
            return

        # CHANGE READ NODE
        self.updateDebugPrint(node, cfile, ufile)
        self.setFilename(node, ufile)
        return node

    def nodeProc(self):
        updatedNodes = []

        data = GetReadNodesData()
        for node in data:
            img = data[node]
            print self.uimg.ipath, img.ipath

            if self.uimg.ipath == img.ipath:

                if self.uimg.layers:
                    if img.clayer in self.uimg.layers:
                        ulimg = self.uimg.layerData[img.clayer]
                        if ulimg.sublayers:
                            if img.csublayer in ulimg.sublayers:
                                setfilename = ulimg.getFile(img.csublayer, img.cctx, img.extension)
                                if self.updateProc(node, img.filename, setfilename):
                                    updatedNodes.append(nuke.toNode(node))
                        else:
                            setfilename = ulimg.getFile('', img.cctx, img.extension)
                            if self.updateProc(node, img.filename, setfilename):
                                updatedNodes.append(nuke.toNode(node))
                else:
                    if self.uimg.sublayers:
                        if img.csublayer in self.uimg.sublayers:
                            setfilename = self.uimg.getFile(img.csublayer, img.cctx, img.extension)
                            if self.updateProc(node, img.filename, setfilename):
                                updatedNodes.append(nuke.toNode(node))
                    else:
                        setfilename = self.uimg.getFile('', img.cctx, img.extension)
                        if self.updateProc(node, img.filename, setfilename):
                            updatedNodes.append(nuke.toNode(node))

        return updatedNodes

if __name__ == '__main__':
    descr = __doc__.strip()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     description=descr)
    parser.add_argument(
        '-n', '--nukeScript', default='',
        help='precomped nuke script'
    )
    parser.add_argument(
        '-i', '--imagePath', default='',
        help='update image version path'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='print update status'
    )
    args = parser.parse_args()

    args.imagePath = PrefixRemove(args.imagePath)

    if args.verbose:
        print '#-----------------------------#'
        print ' >> Input Nuke-script         : ', args.nukeScript
        print ' >> Update Image version-path : ', args.imagePath
        print '#-----------------------------#'

    up = ImageUpdate(args)
    sys.exit(0)
