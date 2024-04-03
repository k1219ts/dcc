'''
exec cmd : DCC.local dev otiotoolkit -- python TimeCodeParser.py
'''
import os
# import scandir
import OpenImageIO as oiio
import pprint

plateDir = '/stuff/prat2/scan/201125/haejeok 002_S010_CG Plate_EDL_201118/002_A046C007_200918_R74I'

info = {}

# for plateDir in os.listdir(platesDir):
directory = os.path.join(plateDir)
plateFile = sorted(os.listdir(directory))[0]
extension = plateFile.split('.')[-1]
# if extension == 'dpx':
#     print "DPX"
# elif extension == 'exr':
#     print "EXR"

img = oiio.ImageInput.open(os.path.join(directory, plateFile))
# print dir(img.spec())
print img.spec().width, img.spec().height
attrs = img.spec()
info[plateDir] = {}
for i in attrs.extra_attribs:
    info[plateDir][i.name] = {'type':i.type, 'value':i.value}

# pprint.pprint(info)