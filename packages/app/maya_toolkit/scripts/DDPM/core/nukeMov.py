import os
import sys
import nuke, nukescripts
import subprocess

platePath = sys.argv[1]
startFrame = sys.argv[2]
endFrame = sys.argv[3]
shotName = sys.argv[4]
project = sys.argv[5]
USERNAME = sys.argv[6]
isRetime = sys.argv[7]
isSound = sys.argv[8]

plateDir = os.path.dirname(platePath)
plateFile = os.path.basename(platePath)

platePathSplit = platePath.split( os.sep )
shotIndex = platePathSplit.index("shot") + 2
shotPath = os.sep.join( platePathSplit[:shotIndex+1] )
renderScriptRoot = os.sep.join( [shotPath, "ani", "dev", "preview"] )
retimeScriptPath = "{0}/comp/retime/script/".format( shotPath )

if os.path.isdir(retimeScriptPath):
    nkList = os.listdir( retimeScriptPath )
    nkFileList = list()

    for i in nkList:
        if i.endswith("nk"):
            nkFileList.append(i)

    SCRIPT = "{0}/comp/retime/script/{1}".format(shotPath, nkFileList[-1])

    timewarpNode = None
    nuke.scriptOpen(SCRIPT)

    AllNodes = nuke.allNodes()

    for node in AllNodes:
        if 'TimeWarp' in node.name():
            timewarpNode = node
else:
    nuke.root()['first_frame'].setValue(int(startFrame))
    nuke.root()['last_frame'].setValue(int(endFrame))

plateNode = nuke.nodes.Read(file=platePath + ".####.jpg", first=startFrame, last=endFrame)
stampNode = nuke.createNode('stamp_log_ani')

if isRetime and os.path.isdir(retimeScriptPath):
    timewarpNode.setInput(0, plateNode)
    stampNode.setInput(0, timewarpNode)
else:
    stampNode.setInput(0, plateNode)

stampNode['Artist_name'].setValue(USERNAME)
stampNode['Shotname'].setValue(shotName)
stampNode.node('P_INPUT1')['message'].setValue('')
stampNode['Project_name'].setValue(project)

writeNode = nuke.nodes.Write(file="{0}_STAMP/{1}.####.jpg".format(plateDir, plateFile),
                             file_type='jpeg')
writeNode['_jpeg_quality'].setValue(1)
writeNode['_jpeg_sub_sampling'].setValue('4:2:2')
writeNode.setInput(0, stampNode)

nukescripts.clear_selection_recursive()

renderScriptPath = '{0}/{1}.nk'.format(renderScriptRoot, shotName)

if not(os.path.exists(os.path.dirname(renderScriptPath))):
    os.makedirs(os.path.dirname(renderScriptPath))
nuke.scriptSaveAs(renderScriptPath, overwrite=1)

writePath = writeNode['file'].value()

if not (os.path.exists(os.path.dirname(writePath))):
    os.makedirs(os.path.dirname(writePath))

nuke.execute(writeNode.name(),
             int(nuke.root()['first_frame'].value()),
             int(nuke.root()['last_frame'].value()))

movOutPath = os.path.dirname(writePath) + '.mov'
ffCmd = ['/backstage/bin/DCC', 'rez-env', 'ffmpeg-4.2.1', '--', 'ffmpeg', '-r', '24', '-start_number',
         str(nuke.root()['first_frame'].value()), '-i', writePath ]
ffCmd += ['-r', '24']
if isSound:
    ffCmd += ['-i', isSound]
else:
    ffCmd += ['-an']
ffCmd += ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'slow']
ffCmd += ['-profile:v', 'baseline', '-b', '6000k', '-tune', 'zerolatency', '-y', movOutPath]

subprocess.call(ffCmd)