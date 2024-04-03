import os
import ice

iFLOAT = ice.constants.FLOAT
currentDir = os.path.dirname(__file__)


def linToRec709(iceImage):
    c1 = ice.Card(iFLOAT, [1.099])
    c2 = ice.Card(iFLOAT, [0.099])
    c3 = ice.Card(iFLOAT, [4.5])
    cp = ice.Card(iFLOAT, [0.45])
    t1 = iceImage.Pow(cp).Multiply(c1).Subtract(c2)
    t2 = iceImage.Multiply(c3)
    base = ice.Card(iFLOAT, [0.018])
    t3 = t1.Multiply(iceImage.Gt(base))
    t4 = t2.Multiply(iceImage.Le(base))
    result = t3.Add(t4)
    return result

def ExrToJpg(filePath):
    newFileName = filePath.replace(".exr", ".jpg")
    loadImg = ice.Load(filePath)
    metadata = loadImg.GetMetaData()
    orgData = metadata["Original Size"].split(' ')
    size = (int(orgData[0][1:]), int(orgData[1][:-1]))
    box = [0, size[0]-1, 0, size[1]-1]
    color = [0.0, 0.0, 0.0, 1.0]
    bg = ice.FilledImage(ice.constants.FLOAT, box, color)
    loadImg = linToRec709(loadImg)
    loadImg = bg.Over(loadImg)
    loadImg.Save(newFileName, ice.constants.FMT_JPEG)

# /backstage/dcc/DCC rez-env prmantoolkit renderman-23.5 oiio ocio_configs -- exr2jpg