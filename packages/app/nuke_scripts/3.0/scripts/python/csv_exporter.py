import csv
import nuke


def csvExPanel():
    nPanel = nuke.Panel("csv File Path")
    nPanel.addFilenameSearch("csv_File_Path : ", "")
    retVar = nPanel.show()
    nameVar = nPanel.value("csv_File_Path : ")
    return (retVar, nameVar)


def csvFileMaker():
    panelValue = csvExPanel()
    if panelValue[0] == 1 and panelValue[1] != '':
        csvName = panelValue[1]

    junkList = []
    shotList = []
    a = nuke.selectedNodes('Read')
    try:
        for n in a:
            b = n.knob('file').value()
            c = b.split('/')[5]
            junkList.append(c)
            for i in junkList:
                if i not in shotList:
                    shotList.append(i)

    except:
        pass

    shotList.sort()


    try:
        listFile = open(csvName, 'w')
        listFile.close()

        for entries in shotList:
            listFile = open(csvName, 'a')
            listFile.write(entries)
            listFile.write('\n')
            listFile.close()
    except:
        pass
