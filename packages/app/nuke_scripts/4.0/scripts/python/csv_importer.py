import csv
import sys, os
import nuke


def csvPanel():
    nPanel = nuke.Panel("cvs File Path")
    nPanel.addFilenameSearch("cvs_File_Path : ", "")
    nPanel.addSingleLineInput("Project_Name : ", "")
    retVar = nPanel.show()
    nameVar = nPanel.value("cvs_File_Path : ")
    prjVar = nPanel.value("Project_Name : ")
    return (retVar, nameVar, prjVar)


def csv_importer():
    panelValue = csvPanel()
    if panelValue[0] == 1 and panelValue[1] != '' and panelValue[2] != '':
        csvName = panelValue[1]
        prjName = panelValue[2]

    try:
        mycsv = csv.reader(open(csvName))
        mycsv.next()

        xposValue = 100
        xpos = 0
        ypos = 0
        a = dict()
        for row in mycsv:
            shotList = row[1]

            seqName = shotList.split('_')[0]
            basePath = "/show/" + prjName + "/shot/" + seqName + '/'
            a = [shotList]
            print(a)
            print(basePath)

            for shotName in a:

                shotPath = os.path.join(basePath, shotName)

                if (os.path.isdir(shotPath)) and ('_' in shotName):
                    platesPath = os.path.join(shotPath, 'plates')
                    if os.path.exists(platesPath):
                        platesDir = os.listdir(platesPath)

                        for plateName in platesDir:
                            if 'retime' in plateName:
                                continue
                            elif plateName.startswith('.'):
                                continue
                            absPlatePath = os.path.join(platesPath, plateName)
                            versionDir = sorted(os.listdir(absPlatePath))

                            if 'lo' in versionDir[0]:
                                versionDir.pop(0)
                            if 'png' in versionDir[0]:
                                versionDir.pop(0)
                            if 'jpg' in versionDir[0]:
                                versionDir.pop(0)
                            for pv in versionDir:
                                if pv.startswith('.'):
                                    continue
                                hvPath = absPlatePath + '/' + pv

                                files = os.listdir(hvPath)
                                firstFile = files[0]
                                ext = firstFile.split('.')[-1]

                                if files:
                                    rg = nuke.nodes.Read()
                                    files.sort()
                                    fileInfo = files[0].split('.')

                                    if '.dpx' in files[0]:
                                        paddingCount = len(files[0].split('.')[1])

                                    else:
                                        paddingCount = len(files[1].split('.')[1])

                                    fullFilePath = hvPath + '/' + fileInfo[0] + '.' + '%04d' + '.' + ext

                                    rg['file'].setValue(fullFilePath)

                                    try:
                                        rg['first'].setValue(int(fileInfo[1]))
                                        rg['last'].setValue(int(fileInfo[1]) + len(files) - 1)
                                    except:
                                        pass

                                    rg.setXYpos(xpos, ypos)
                                    ypos -= 100

                    nameNode = nuke.nodes.StickyNote()
                    nameNode['note_font_size'].setValue(20)
                    nameNode.setXYpos(xpos, 100)
                    nameNode['label'].setValue(shotName)
                    print("")
                    #                        dot = nuke.nodes.dot()
                    #                        dot.setXYpos(xpos, 0)

                    xpos += 200
                    ypos = 0
            ypos += 500
    except:
        pass
