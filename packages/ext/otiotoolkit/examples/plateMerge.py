import os
import shutil

srcDir = '/stuff/emd/scan/20210205_2/210201_re_vfx/Emergency_C09_12th_CGSource_210126/372_A063C016_200611_R4AK'
dstDir = '/stuff/emd/scan/20210205_2/210201_re_vfx/Emergency_C09_12th_CGSource_210126/373_A063C016_200611_R4AK'

srcLastFile = sorted(os.listdir(srcDir))[-1]
srcFrameStr = srcLastFile.split('.')[-2]
framePadding = len(srcFrameStr)

print srcFrameStr
for index, imgFile in enumerate(sorted(os.listdir(dstDir))):
    splitImgFile = imgFile.split('.')
    splitImgFile[-2] = str((int(srcFrameStr) + (index + 1))).zfill(framePadding)
    newImgFile = os.path.join(srcDir, '.'.join(splitImgFile))
    orgImgFile = os.path.join(dstDir, imgFile)

    print orgImgFile, '->', newImgFile

    shutil.copy2(orgImgFile, newImgFile)