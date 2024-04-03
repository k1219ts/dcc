import hou
import os
import json
import string



def getAlembicFileName(filename, objtype):
    if objtype == 'high':
        object_type = ''
    else:
        object_type = objtype

    filePath = os.path.dirname(filename)
    baseName, extension = os.path.splitext(os.path.basename(filename))

    result = filename

    source = baseName.split('_')
    for t in ['low', 'mid', 'sim']:
        if t in source:
            source.remove(t)

    if len(source) == 1:
        temp = list(source)
        if object_type:
            temp.insert(1, object_type)
        fn = os.path.join(filePath, string.join(temp, '_') + extension)
        if os.path.exists(fn):
            return fn
    else:
        for i in range(1, len(source)+1):
            temp = list(source)
            if object_type:
                temp.insert(i, object_type)
            fn = os.path.join(filePath, string.join(temp, '_') + extension)
            if os.path.exists(fn):
                return fn

    return result




class ZENV:
    def __init__(self, filename):
        self.m_data = dict()

        f = open(filename, 'r')
        data = json.load(f)
        if data.has_key('InstanceSetup'):
            self.m_data = data['InstanceSetup']
        f.close()


    def doIt(self):
        if not self.m_data:
            return

        # create Geometry ZENV
        self.m_zenv = hou.node('/obj/ZENV')
        if not self.m_zenv:
            self.m_zenv = hou.node('/obj').createNode('geo', 'ZENV')
            self.m_zenv.setColor(hou.Color(1.0, 0.725, 0.0))
            hou.node(self.m_zenv.path()+'/file1').destroy()

        # create Alembic ZENV/SOURCE
        if self.m_data.has_key('abcfiles'):
            print '#' * 50
            print '#'
            print '# SOURCE Setup'
            print '#'
            print '#' * 50

            self.m_source = hou.node('/obj/ZENV/SOURCE')
            if not self.m_source:
                self.createSourceSubNet()
            # source alembic
            for fn in self.m_data['abcfiles']:
                self.createZenvSourceGroups(self.m_source, fn)

            self.setSourceObjectPathParameters()

            # layout source
            self.layout_SOURCE_children()

        # create Assembly Points ZENV/ASB_NAME
        if self.m_data.has_key('asbfiles'):
            print '#' * 50
            print '#'
            print '# Points Group Setup'
            print '#'
            print '#' * 50

            merge = hou.node('/obj/ZENV/points_merge')
            if merge:
                clearInputConnections(merge)
            else:
                merge = self.m_zenv.createNode('merge', 'points_merge')

            for i in range(len(self.m_data['asbfiles'])):
                fn = self.m_data['asbfiles'][i]
                pointGroup = self.createPointsGroup(fn)
                self.layout_PointsGroup_children(pointGroup)
                merge.setInput(i, pointGroup)

                # parameter link
                if not pointGroup.parm('viewportlod'):
                    parm_grp = pointGroup.parmTemplateGroup()
                    src_copy = hou.node(pointGroup.path() + '/src_copy')
                    parmTemp = src_copy.parm('viewportlod').parmTemplate()
                    parm_grp.append(parmTemp)
                    pointGroup.setParmTemplateGroup(parm_grp)
                    pointGroup.parm('viewportlod').set(1)
                    src_copy.parm('viewportlod').setExpression(
                        'ch("../viewportlod")', hou.exprLanguage.Hscript, True
                    )


            output = hou.node('/obj/ZENV/output')
            if not output:
                output = self.m_zenv.createNode('output', 'output')
                output.setInput(0, merge)

        # layout zenv
        self.layout_ZENV_children()



    # Create SOURCE
    def createSourceSubNet(self):
        self.m_source = self.m_zenv.createNode('subnet', 'SOURCE')
        self.m_source.setColor(hou.Color(0.094, 0.369, 0.69))
        hideSubnetLabel(self.m_source)
        self.parameterSetupForSourceSubNet()
        self.m_source.createNode('subnet', 'null')

    #   source parameter setup
    def parameterSetupForSourceSubNet(self):
        before = 'label4'
        parm_grp = self.m_source.parmTemplateGroup()

        otypeParm = hou.MenuParmTemplate('objtype', 'Object Type',
                                         join_with_next=True,
                                         menu_items=(['low', 'mid', 'high', 'sim']),
                                         menu_labels=(['Low', 'Mid', 'High', 'Simul']),
                                         default_value=0,
                                         menu_type=hou.menuType.Normal)
        callbackScript = objectTypeCallbackScript()
        otypeParm.setScriptCallback(callbackScript)
        otypeParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
        otypeParm.setTags(
            {'script_callback': callbackScript, 'script_callback_language': 'python'}
        )
        parm_grp.insertAfter(before, otypeParm)

        viewParm = hou.ButtonParmTemplate('srcview', 'All Hide')
        callbackScript = 'import zenvjson\nzenvjson.displayOffAll()\n'
        viewParm.setScriptCallback(callbackScript)
        viewParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
        parm_grp.insertAfter(otypeParm, viewParm)

        before = viewParm

        for p in self.m_data['locations']:
            id = self.m_data['locations'].index(p) + 100

            setlabels = tuple(p.split('/'))
            if len(setlabels) < 3:
                setlabels = setlabels + ('', '')
            labelParm = hou.LabelParmTemplate('srclabel%s' % id, 'label',
                                              is_label_hidden=True,
                                              column_labels=setlabels)
            parm_grp.insertAfter(before, labelParm)

            viewParm = hou.ButtonParmTemplate('srcview%s' % id, 'view',
                                              join_with_next=True)
            callbackScript = displayCallbackScript('srcpath%s' % id)
            viewParm.setScriptCallback(callbackScript)
            viewParm.setScriptCallbackLanguage(hou.scriptLanguage.Python)
            viewParm.setTags(
                {'script_callback': callbackScript, 'script_callback_language': 'python'}
            )
            parm_grp.insertAfter(labelParm, viewParm)

            pathParm = hou.StringParmTemplate('srcpath%s' % id, '', 1,
                                              is_label_hidden=True,
                                              string_type=hou.stringParmType.NodeReference)
            parm_grp.insertAfter(viewParm, pathParm)

            before = pathParm

        self.m_source.setParmTemplateGroup(parm_grp)


    # set SOURCE object path
    def setSourceObjectPathParameters(self):
        for i in range(len(self.m_data['locations'])):
            p = self.m_data['locations'][i]
            src = p.split('/')
            objpath = self.m_source.path() + '/' + src[0]
            if len(src) > 1:
                objpath += '/' + '_' + string.join(src[1:], '_')
            objpath_node = hou.node(objpath)
            if objpath_node.type().name() == 'subnet':
                objpath += '/' + src[0]
            self.m_source.parm('srcpath%s' % (i+100)).set(objpath)



    # Create ZenvSource Group
    def createZenvSourceGroups(self, parent, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        # subnet check
        pathList = list()
        for p in self.m_data['locations']:
            if p.find(basename) > -1:
                gname = p.replace(basename, '')
                if gname:
                    pathList.append(gname)

        source = hou.node(parent.path() + '/' + basename)
        if source:
            print '>> Source Created'
            print '\t[file] : %s' % filename
            if pathList:
                sourceGroup = source
                sourceNode  = hou.node(sourceGroup.path() + '/' + basename)
                self.sourceObjectPathSetup(sourceGroup, sourceNode, pathList)
                self.layout_ZenvSourceGroup_children(sourceGroup)
        else:
            print '>> Create Source'
            print '\t[file] : %s' % filename
            if pathList:
                sourceGroup = parent.createNode('subnet', basename)
                hideSubnetLabel(sourceGroup)
                sourceNode  = self.alembicCreateNode(sourceGroup, filename)
                self.sourceObjectPathSetup(sourceGroup, sourceNode, pathList)
                self.layout_ZenvSourceGroup_children(sourceGroup)
            else:
                sourceNode = self.alembicCreateNode(parent, filename)


    def alembicCreateNode(self, parent, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        abcNode = hou.node(parent.path() + '/' + basename)
        if not abcNode:
            abcNode = parent.createNode('alembic', basename)
        # first create, low object type set
        abcNode.parm('fileName').set(getAlembicFileName(filename, 'low'))
        abcNode.parm('loadmode').set(2)
        abcNode.parm('groupnames').set(2)
        return abcNode


    # pathsource : ['/../..', '/../../..']
    def sourceObjectPathSetup(self, parent, abcnode, pathsource):
        print '\t>> Object Path'
        for p in pathsource:
            print '\t\tpath : %s' % p
            node_name = p.replace('/', '_')
            objnode = hou.node(parent.path() + '/' + node_name)
            if not objnode:
                objnode = parent.createNode('delete', node_name)
            objnode.setInput(0, abcnode)
            objnode.parm('group').set('*' + p.split('/')[-1] + '*')
            objnode.parm('negate').set(1)

        if len(pathsource) == 1:
            objnode.parm('group').set('*')



    # Create PointsGroup
    def createPointsGroup(self, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        pointGroup = hou.node(self.m_zenv.path() + '/' + basename)
        if not pointGroup:
            print '>> Create Points Group'
            pointGroup = self.m_zenv.createNode('subnet', basename)
            hideSubnetLabel(pointGroup)
        else:
            print '>> Points Group Created'
        print '\t[file] : %s' % filename

        pointNode = hou.node(pointGroup.path() + '/' + basename)
        if not pointNode:
            print '\t>> Create PointsAlembic'
            pointNode = pointGroup.createNode('alembic', basename)
        pointNode.parm('fileName').set(filename)
        pointNode.parm('loadmode').set(2)
        pointNode.parm('reload').pressButton()

        switch = self.createPointsSourceObject(pointGroup, pointNode)

        copy = hou.node(pointGroup.path() + '/' + 'src_copy')
        if not copy:
            copy = pointGroup.createNode('copy', 'src_copy')
        copy.parm('stamp').set(1)
        copy.parm('cacheinput').set(1)
        copy.parm('pack').set(1)
        copy.parm('viewportlod').set(1)
        copy.parm('param1').set('rtp')
        copy.parm('val1').setExpression('@rtp')

        switch.parm('input').setExpression(
            'stamp("../%s", "rtp", 0)' % copy.name()
        )

        copy.setInput(0, switch)
        copy.setInput(1, pointNode)

        output = hou.node(pointGroup.path() + '/' + 'output')
        if not output:
            output = pointGroup.createNode('output', 'output')
            output.setInput(0, copy)

        return pointGroup


    def createPointsSourceObject(self, parent, asbnode):
        geo = asbnode.geometry()
        arcfiles = ''; arcpath = ''
        if geo.findPrimAttrib('arcfiles'):
            arcfiles = geo.primStringAttribValues('arcfiles')
        if not arcfiles:
            return

        if geo.findPrimAttrib('arcpath'):
            arcpath = geo.primStringAttribValues('arcpath')

        switch = hou.node(parent.path() + '/' + 'src_switch')
        if switch:
            clearInputConnections(switch)
        else:
            switch = parent.createNode('switch', 'src_switch')

        print arcfiles, arcpath

        for i in range(len(arcfiles)):
            basename = os.path.splitext(os.path.basename(arcfiles[i]))[0]
            name = basename
            objpath = '/obj/ZENV/SOURCE/' + basename
            # if arcpath:
            #     hpath = arcpath[i].replace('/', '_')
            #     name = hpath
            #     objpath += '/' + hpath

            print objpath

            objpath_node = hou.node(objpath)
            if objpath_node.type().name() == 'subnet':
                objpath += '/' + basename

            object_merge = hou.node(parent.path() + '/' + name)
            if not object_merge:
                object_merge = parent.createNode('object_merge', name)
            object_merge.parm('objpath1').set(objpath)

            switch.setInput(i, object_merge)

        return switch



    # subnet layout
    def layout_ZENV_children(self):
        voffset = -1
        others = list()
        for n in self.m_zenv.children():
            if n.type().name() == 'subnet':
                if n.name() == 'SOURCE':
                    n.setPosition(hou.Vector2(0, 0))
                else:
                    n.setPosition(hou.Vector2(0, voffset))
                    voffset -= 1
            else:
                others.append(n)

        voffset = voffset / 2
        for n in others:
            n.setPosition(hou.Vector2(10, voffset))
            voffset -= 1

    def layout_SOURCE_children(self):
        subnets = list()
        abcs = list()
        for n in self.m_source.children():
            if n.type().name() == 'subnet':
                subnets.append(n)
            else:
                abcs.append(n)

        gx, gy = self.getStartPosition(self.m_source)

        # subnets
        for i in range(len(subnets)):
            x = i / 10
            y = i - (i / 10) * 10
            x = x*5; y *= -1
            x += gx; y += gy
            subnets[i].setPosition(hou.Vector2(x, y))
        # abcs
        for i in range(len(abcs)):
            x = i / 10
            y = i - (i / 10) * 10
            x = x*5 + len(subnets)/10*5 + 5
            y *= -1
            x += gx; y += gy
            abcs[i].setPosition(hou.Vector2(x, y))

    def layout_ZenvSourceGroup_children(self, subnet):
        gx, gy = self.getStartPosition(subnet)

        abcs   = list()
        others = list()
        for n in subnet.children():
            if n.type().name() == 'alembic':
                abcs.append(n)
            else:
                others.append(n)

        for i in range(len(abcs)):
            x = gx
            y = gy - len(others)/2 - i
            abcs[i].setPosition(hou.Vector2(x, y))

        for i in range(len(others)):
            x = gx + 5
            y = gy - i
            others[i].setPosition(hou.Vector2(x, y))

    def layout_PointsGroup_children(self, subnet):
        gx, gy = self.getStartPosition(subnet)

        object_merges = list()
        switch = ''
        points = ''
        copy = ''
        output = ''

        for n in subnet.children():
            type_name = n.type().name()
            if type_name == 'object_merge':
                object_merges.append(n)
            elif type_name == 'switch':
                switch = n
            elif type_name == 'alembic':
                points = n
            elif type_name == 'copy':
                copy = n
            else:
                output = n

        for i in range(len(object_merges)):
            object_merges[i].setPosition(hou.Vector2(gx, gy-i))

        sx = gx + 10
        sy = gy - len(object_merges)/2
        switch.setPosition(hou.Vector2(sx, sy))

        points.setPosition(hou.Vector2(sx+2, sy))

        copy.setPosition(hou.Vector2(sx+1, sy-1))
        output.setPosition(hou.Vector2(sx+1, sy-3))


    def getStartPosition(self, subnet):
        inputs = subnet.indirectInputs()
        x, y = inputs[0].position()
        return x, y-1



# Display Control
def displayCallbackScript(parm_name):
    callbackScript = '''
import zenvjson
zenvjson.displayOnCurrentNode('%s')
''' % parm_name
    return callbackScript


def displayOffCurrentNode():
    node = hou.pwd()
    for parm in node.parms():
        if parm.name().find('srcpath') > -1:
            objpath = parm.eval()
            objnode = hou.node(objpath)
            objnode.setDisplayFlag(False)
            if objnode.type().name() == 'delete':
                parent = objnode.parent()
                parent.setDisplayFlag(False)


def displayOnCurrentNode(parm_name):
    node = hou.pwd()
    displayOffCurrentNode()
    objpath = node.parm(parm_name).eval()
    objnode = hou.node(objpath)
    if objnode.type().name() == 'delete':
        parent = objnode.parent()
        parent.setDisplayFlag(True)
    objnode.setDisplayFlag(True)


def displayOffAll():
    node = hou.pwd()
    null = hou.node(node.path() + '/null')
    null.setDisplayFlag(True)
    displayOffCurrentNode()



# Object Type Change
def objectTypeCallbackScript():
    callbackScript = '''
import zenvjson
zenvjson.changeZenvSourceObject()
'''
    return callbackScript

def changeZenvSourceObject():
    node = hou.pwd()
    objtype = node.parm('objtype').evalAsString()

    abcnodes = list()
    for n in node.allSubChildren():
        if n.type().name() == 'alembic':
            abcnodes.append(n)

    print '>> Source Object Change'
    for n in abcnodes:
        fn = n.parm('fileName').eval()
        new = getAlembicFileName(fn, objtype)
        print '\t: %s' % new
        n.parm('fileName').set(new)
        n.parm('reload').pressButton()




def hideSubnetLabel(subnet):
    subnet.parm('label1').hide(True)
    subnet.parm('label2').hide(True)
    subnet.parm('label3').hide(True)
    subnet.parm('label4').hide(True)

def clearInputConnections(node):
    for i in node.inputConnections():
        node.setInput(0, None)




# Main Process
def import_data(filename):
    zc = ZENV(filename)
    zc.doIt()


def doIt():
    fn = '/show/test_shot/asset/global/user/sanghun/data/scripttest/iceLake.json'
    zc = ZENV(fn)
    zc.doIt()
