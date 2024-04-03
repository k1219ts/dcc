### Linux batch v0.3
### Last modified: 2009/07/28
### Written by Kim Dong Su
### defucon7@naver.com
import nuke
import re
import os
import subprocess

def linuxBatch_global():
    a = nuke.knob("first_frame")
    b = nuke.knob("last_frame")
    start = int(a)
    end = int(b)
    incr = 1
    _range = a+","+b
    r = nuke.getFramesAndViews("Frames to execute:", _range)
    f = re.sub("[,|-|/]", " ", r[0])
    t = f.split()
    if len(t) > 0:
        start = int(t[0])
        end = start
    if len(t) > 1: end = int(t[1])
    if len(t) > 2: incr = int(t[2])


    """
    Creates a batch file from selected Write nodes in the DAG
    """

    ### Custom Options #########################################################
    shellFlags = "-X"
    savePath = "/tmp/"
    ##########################################################################

    nMajor = nuke.NUKE_VERSION_MAJOR
    nMinor = nuke.NUKE_VERSION_MINOR
    nRelease = nuke.NUKE_VERSION_RELEASE
    #nDir = "/usr/local/Nuke6.3v8/Nuke"
    nDir = "/usr/local/Nuke" + nuke.NUKE_VERSION_STRING + '/Nuke'
    nVersion = str(nDir) + str(nuke.NUKE_VERSION_MAJOR) + "." + str(nuke.NUKE_VERSION_MINOR)
    scriptPath = nuke.Root().name()
    batPath = savePath + scriptPath.split('/').pop().rstrip('.nk') + ".sh"

    # Check Nuke version
    if nMajor <= 5 and nMinor < 1:
        nuke.message("This script only runs in Nuke6.2v1 or higher")

    else:
        sn = nuke.selectedNodes('Write')
        renderlist = []

        if sn == []:
            nuke.message("No write nodes selected")

        else:
            if scriptPath == "":
                nuke.message("You must save your script first")
                nuke.scriptSave()
                ask = nuke.ask("Script is now saved\nContinue?")
                if ask == True:
                    linuxBatch_global()
                else:
                    nuke.tprint("Generate Batch was aborted")

            else:
                try:
                    nuke.scriptSave()
                    nuke.tprint("Script is now saved")

                except Exception:
                    nuke.message("The Script couldn't be saved")

                for n in sn:
                    tl = []
                    tl.append(int(n.knob('render_order').value()))
                    tl.append(n.name())
                    tl.append(start)
                    tl.append(end)
                    renderlist.append(tl)

                try:
                    bat = open(batPath, 'w')
                    bat.write("#!/bin/bash \n")
                    bat.write("# \n")
                    os.chmod(batPath, 0o777)
                    for each in sorted(renderlist):
                        bat.write(str(nVersion) + " " + str(shellFlags) + " " + str(each[1]) + " " + "\"" + str(scriptPath) + "\"" + " " + str(each[2]) + "," + str(each[3]) + "\n")
                    bat.write("\n echo -n ------------------ Render Complete \n")
                    bat.write("read dd \n")
                    bat.close()
                except EnvironmentError:
                    nuke.message("Can't save a batch file!\nCheck the savePath variable in the linuxBatch_global.py script Custom Options\nOr check the target file/folder access rights")

                ask = nuke.ask("Start shell render now?")

                if ask == True:
                    try:
                        writeNode = nuke.selectedNode()
                        if not (os.path.exists(os.path.dirname(writeNode['file'].value()))):
                            os.makedirs(os.path.dirname(writeNode['file'].value()))
                        os.system('gnome-terminal -x %s'%(batPath))


                    except EnvironmentError:
                        nuke.message("Nuke can't start your batch file\nYour batch file can be found at at:\n" + str(batPath))

                else:
                    nuke.message("Your batch file can be found at at:\n" + str(batPath))
