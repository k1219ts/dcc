import csv
import os
import nuke


def newest(path):
    dirs = sorted(os.listdir(path))
    global last
    for basename in dirs:
        mid = basename.split('_')[2]
        if mid == 'comp':
            paths = [os.path.join(path, basename)]
            last = max(paths, key=os.path.getctime)
        else:
            pass

    return last

def csvPanel():
    nPanel = nuke.Panel("cvs File Path")
    nPanel.addFilenameSearch("cvs_File_Path : ", "")
    nPanel.addSingleLineInput("Project_Name : ", "")
    retVar = nPanel.show()
    nameVar = nPanel.value("cvs_File_Path : ")
    prjVar = nPanel.value("Project_Name : ")
    return (retVar, nameVar, prjVar)

def lastNodeContact():
    #nukescripts.clear_selection_recursive()

    panelValue = csvPanel()
    if panelValue[0] == 1 and panelValue[1] != '' and panelValue[2] != '':
        csvName = panelValue[1]
        prjName = panelValue[2]


    mycsv = csv.reader(open(csvName))
    mycsv.next()
    xposValue = 100
    xpos = 0
    ypos = 0
    a = dict()
    dpxPath = []
    for row in mycsv:
        shotList = row[1]

        seqName = shotList.split('_')[0]
        basePath = "/show/" + prjName + "/shot/" + seqName + '/'

        a = [shotList]

        for shotName in a:
            shotPath = os.path.join(basePath, shotName)
            truePath = shotPath + '/comp/comp/render/dpx/'
            if os.path.exists(truePath) == True:
                # midPath = "/comp/comp/render/dpx/"
                dpxPath = truePath
                fullPath = newest(dpxPath)
                files = os.listdir(fullPath)

                rg = nuke.nodes.Read()
                files.sort()
                fileCount = len([f for f in files if f[-4:] == ".dpx"])
                fileInfo = files[0].split('.')
                paddingCount = len(files[0].split('.')[1])
                fullFilePath = fullPath + '/' + fileInfo[0] + '.' + '%04d' + '.dpx'

                print(fullFilePath)

                rg['file'].setValue(fullFilePath)
                rg['before'].setValue(1)
                rg['after'].setValue(1)
                try:
                    rg['first'].setValue(int(fileInfo[1]))
                    rg['last'].setValue(int(fileInfo[1]) + int(fileCount) - 1)
                except:
                    pass
                rg.setXYpos(xpos, ypos)
                ypos -= 100
            nameNode = nuke.nodes.StickyNote()
            nameNode['note_font_size'].setValue(20)
            nameNode.setXYpos(xpos, -100)
            nameNode['label'].setValue(shotName)
            print("")
            #                        dot = nuke.nodes.dot()
            #                        dot.setXYpos(xpos, 0)
            xpos += 200
            ypos = 0
    ypos += 500
