import maya.standalone
import maya.cmds as cmds
maya.standalone.initialize(name='python')

if not cmds.pluginInfo('glmCrowd', q=True, l=True):
    cmds.loadPlugin('glmCrowd')


import glm.devkit as glmapi


sceneFile = '/show/nxer/works/CRD/shot/ER05/ER05_0120/pub/scenes/ER05_0120_crowd_v001.mb'
cmds.file(sceneFile, o=True, f=True)

getGcha = cmds.getAttr('cacheProxyShape2.characterFiles')
gcha = getGcha.split(';')

glmapi.initGolaem()
glmapi.initSimulationCacheFactory()
glmapi.loadSimulationCacheFactoryCharacters(';'.join(gcha))

charsBoneIds = []
charsParentBoneIds = []

for f in gcha:
    boneNames = glmapi.getBoneNames(f).split(';')
    boneIds = glmapi.intArray_frompointer(glmapi.getSortedBones(f))
    pBoneIds = glmapi.intArray_frompointer(glmapi.getParentBones(f))

    tmpB = []
    tmpPB = []
    for i in range(0, len(boneNames)-1):
        tmpB.append(boneIds[i])
        tmpPB.append(pBoneIds[i])

    charsBoneIds.append(tmpB)
    charsParentBoneIds.append(tmpPB)

boneId = glmapi.intArray_frompointer(glmapi.getSortedBones(gcha[0]))
boneName = glmapi.getBoneNames(gcha[0]).split(';')

# for idx,i in enumerate(boneName):
#     print idx, i

# for i in range(0, 135):
#     print boneId[i]
#
# print '-'*50
#
for i in range(0, 135):
    print charsBoneIds[0][i]

print '-'*50

for i in range(0, 135):
    print charsBoneIds[1][i]
