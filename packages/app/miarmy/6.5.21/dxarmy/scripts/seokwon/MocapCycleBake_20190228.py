# 원본 + 대상 캐릭터의 최상위 노드를 순서대로 선택 후 실행
conList = ['root_CON', 'C_IK_head_CON', 'R_IK_hand_CON', 'L_IK_hand_CON', 'R_IK_foot_CON', 'L_IK_foot_CON']
vecList = ['R_IK_footVec_CON', 'L_IK_footVec_CON', 'R_IK_handVec_CON', 'L_IK_handVec_CON']
sel = cmds.ls(sl=True)
delConList = list()
for i in conList:
    pCon = str(cmds.parentConstraint(str(sel[0].split(":")[0]) + ":" + i, str(sel[-1].split(":")[0]) + ":" + i, mo=False, w=True)[0])
    delConList.append(pCon)
for j in vecList:
    eCon = str(cmds.pointConstraint(str(sel[0].split(":")[0]) + ":" + j, str(sel[-1].split(":")[0]) + ":" + j, mo=False, w=True)[0])
    delConList.append(eCon)

cmds.setAttr(sel[0] + ".visibility", 0)

# Bake
cmds.currentTime(cmds.playbackOptions(q=True, min=True))
bst = conList + vecList + ["move_CON"]
bakeList = list()
for i in bst:
    bakeList.append(str(sel[-1].split(":")[0]) + ":" + i)
cmds.bakeResults(bakeList, simulation=True, t=(cmds.playbackOptions(q=True, min=True), cmds.playbackOptions(q=True, max=True)), sampleBy=1, dic=True, pok=True, sac=False, ral=False, bol=False, mr=True, controlPoints=False, shape=True)


# 컨스트레인 제거
cmds.delete(delConList)
