
import os
import re
import maya.cmds as cmds
import maya.mel as mm
import sgUI
import sgComponent


def setComponentNodeOptions(node, attr, value, **kwargs):
    """

    :type node: str
    :param attr: attribute name
    :param value: attribute value
    :type kwargs: dict
    """
    cmds.setAttr("{0}.{1}".format(node, attr), value, **kwargs)


def importLayoutFileDialog():
    fn = cmds.fileDialog2(fm=4,
                          ff='Alembic (*.abc)',
                          cap='Import Alembic Component (Select File)',
                          okc='import')
    if not fn:
        return

    ciClass = ComponentImport(Files=fn, World="Baked")
    ciClass.m_mode = 0 #meshmode
    ciClass.m_display = 1 #Render
    ciClass.m_fitTime = 0
    nodes = ciClass.doImport()

    if nodes:
        for node in nodes:
            setComponentNodeOptions(node, "action", 2)


class ComponentImport(sgUI.ComponentImport):
    def importAlembic(self, parentNode, filename):
        mm.eval('AbcImport -d -m import -rpr "%s" "%s"' % (parentNode, filename))


    def addVisibilityControl(self, cpClass, filename, componentNode):
        lowFilename = sgComponent.get_reloadFileName(filename, 3)
        arcNode = cpClass.m_arcNode

        self.importAlembic(arcNode, lowFilename)
        arcChilds = cmds.listRelatives(arcNode, c=True)

        lowMeshGRP = str()
        highMeshGRP = str()

        if len(arcChilds) == 2:
            for i in arcChilds:
                if not i.find("_low_") == -1:
                    lowMeshGRP = i
                else:
                    highMeshGRP = i
            cmds.addAttr(componentNode, longName="meshType", at="enum", en="Low:High", k=True)
            conditionNode = cmds.createNode('condition', ss=True)
            cmds.setAttr(conditionNode + ".secondTerm", 1)
            cmds.setAttr(conditionNode + ".colorIfTrueG", 1)
            cmds.setAttr(conditionNode + ".colorIfFalseG", 0)
            cmds.connectAttr(componentNode + ".meshType", conditionNode + ".firstTerm")
            cmds.connectAttr(conditionNode + ".outColorR", lowMeshGRP + ".visibility")
            cmds.connectAttr(conditionNode + ".outColorG", highMeshGRP + ".visibility")
            cmds.setAttr('%s.display' % componentNode, lock=True)


    def doImport(self):
        files = self.getFiles()
        files.sort()

        createNodeList = list()
        for f in files:
            baseName = os.path.basename(f)
            splitVer = re.compile(r'_v\d+.abc').findall(baseName)
            if splitVer:
                nodeName = baseName.split(splitVer[0])[0]
            else:
                nodeName = baseName.split('.abc')[0]

            compoNode = cmds.createNode('dxComponent', n=nodeName)
            createNodeList.append(compoNode)
            cmds.setAttr('%s.abcFileName' % compoNode, f, type='string')
            cmds.setAttr('%s.mode' % compoNode, self.m_mode)
            cmds.setAttr('%s.display' % compoNode, self.m_display)
            if self.m_world:
                worldFile = None
                # old-style world file
                wf = f.replace('.abc', '.world')
                if os.path.exists(wf):
                    worldFile = wf
                # new-style alembic world file
                wf = f.replace('.abc', '.wrd')
                if os.path.exists(wf):
                    worldFile = wf
                if worldFile:
                    cmds.setAttr('%s.worldFileName' % compoNode, worldFile, type='string')

            cpClass = sgComponent.Archive(compoNode)
            if self.m_world == 1:
                cpClass.m_baked = True
            cpClass.doIt()

            #Ani edit
            self.addVisibilityControl(cpClass, f, compoNode)


        return createNodeList