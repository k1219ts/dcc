## WHEN MODIFIERS ARE HELD DOWN ##
## CTRL: Files will be imported with relative references - $JOB/$HIP/$HOME variables are used if found.
## SHIFT: PBR connections for Arnold/RS will be ignored. '.hip' files will allow you merge nodes into your current hip file. 
## '.vdb' files will be brought in as an arnold volume.
## ALT: If your file is a sequence the frame number will be replaced with $F.

import hou,re,os,random,colorsys,mergeTree,sys
from PySide2 import QtCore,QtWidgets
from utils import promoteHip

reload(promoteHip)
reload(mergeTree)

## Format Tuples

hiplist = (('.hip','.hipnc','.hiplc'))
piclist = (('.tx', '.als', '.bmp', '.cin', '.dpx', '.gif', '.dsm', 'jpg', '.exr', '.hdr', '.ies', '.jpeg', '.kdk', '.pic', '.pic.gz', '.pic.Z', '.picgz', '.piclc', '.picmc', '.picZ', '.pix', '.png', '.psb', '.psd', '.ptex', '.ptx', '.qtl', '.rat', '.rgb', '.rgba', '.rla', '.rla16', '.rlb', '.rlb16', '.sgi', '.su', '.tbf', '.tga', '.tif', '.tif16', '.tif3', '.tif32', '.tiff', '.vst', '.vtg', '.yuv'))
geolist = (('.abc', '.ai', '.bgeo', '.bgeo.bz2', '.bgeo.lzma', '.bgeo.sc', '.bgeogz', '.bgeo.gz', '.bgeosc', '.bhclassic', '.bhclassic.bz2', '.bhclassic.gz', '.bhclassic.lzma', '.bhclassic.sc', '.bhclassicgz', '.bhclassicsc', '.bjson', '.bjson.sc', '.bjsongz', '.bjsonsc', '.bpoly', '.bstl', '.d', '.dxf', '.eps', '.fbx', '.geo', '.geo.gz2', '.geo.gz', '.geo.lama', '.geo.sc', '.geogz', '.geosc', '.hclassic', '.hclassic.bz2', '.hclassic.gz', '.hclassic.lzma', '.hclassic.sc', '.iges', '.igs', '.json', '.json.gz', '.json.sc', '.jsongz', '.jsonsc', '.lw', '.lwo', '.obj', '.off', '.pc', '.pbd', '.ply', '.pmap', '.poly', '.rib', '.stl', '.vdb'))
otllist = (('.otl','.hda'))
asslist = (('.ass',))
redlist = (('.rs',))
usdlist = (('.usd',))

allFormats = hiplist + piclist + otllist + geolist + asslist + redlist + usdlist

## Enviroment Variables

envVariables = (('$JOB','$HIP','$HOME'))

## Get Houdini Major Release

houVer = int(hou.expandString('$HOUDINI_VERSION').split('.',1)[0])

def dropAccept(file_list):
    modifiers()
    getExtension(file_list)

    if sys.platform != 'win32':
        begin = '/'
    else:
        begin = ''
    
    file_list = ['%s%s'%(begin,f.split('file:///',1)[1]) if f.startswith('file:///') else f for f in file_list]
    file_list = expandCheck(file_list)
    if Alt:
        file_list = seqCheck(file_list)
    
    if ext in hiplist:
        if ext == '.hip' and Shift:
            mergeTree.run(file_list[0])

        elif Shift:
            if file_list[0].find(' ') != -1:
                mergeTree.run(file_list[0])

            else:
                convertHip(file_list)

        else:
            hou.hipFile.load(file_list[0], False, False)
    
    elif ext in piclist:
        pic(file_list)

    elif ext in geolist:
        geo(file_list)

    elif ext in otllist:
        otl(file_list)

    elif ext in asslist:
        prox(file_list,0)

    elif ext in redlist:
        prox(file_list,1)

    elif ext in usdlist:
        usd(file_list)

    return True

def modifiers():
    global Ctrl,Alt,Shift
    Ctrl = (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier
    Alt = (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier
    Shift = (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier

def getExtension(file_list):
    global ext
    try:
        ext = [ext for ext in allFormats if file_list[0].lower().endswith(ext)][0]
    except:
        return True

def expandCheck(file_list):
    #if not hou.ui.displayMessage('Use $HIP/$HOME/$JOB in the path?', buttons=("Yes","No"), severity=hou.severityType.Message, title="Environment Variable Found in Path",close_choice=1):
    if Ctrl:
        templist = []
        for file in file_list:
            for v in envVariables:
                expanded = hou.expandString(v)
                file = file.replace(expanded,v)

            templist.append(file)

            file_list = templist

#    for n,p in os.environ.items():
#        p = p.replace('\\','/')
#        if '/' in p:
#            if p in file_list[0]:
#                file_list[0] = path.replace(p,'$'+n)

    return file_list

def seqCheck(file_list):
    templist = []

    for f in file_list:
        path,filename = f.rsplit('/',1)

        sep = None

        formats = ['\.','_']
        for fmt in formats:
            if len(re.findall('%s(?=\d+)'%fmt, filename)) > 0:
                splt = re.split('%s(?=\d+)'%fmt, filename)
                sep = fmt
                string,frame = [''.join(splt[:-1]),splt[-1]]
                frame,end = re.split(r'(^[^\D]+)', frame)[1:]
                break
                
        if sep:

            for file in os.listdir(path):
                if file.startswith(string) and file.endswith(end):
                    file = file[:-len(end)]
                    if file != filename:
                        if len(re.findall('%s(?=\d+)'%sep, file)) > 0:
                            padding = len(re.split('%s(?=\d+)'%sep, file)[-1])
                            if padding == 1: padding = ''
                            else: padding = str(padding)

                            if sep.startswith('\\'): sep = sep[1:]
                            f = '%s/%s%s$F%s%s'%(path,string,sep,padding,end)
                            break

        templist.append(f)

    file_list = templist

    return file_list

def otl(file_list):
    cursor(file_list)
    randcol()
    create = True
    if not parentnode: create = False
    try: pos
    except: create = False

    for f in file_list:
        if not f.endswith(otllist): continue

        defin = hou.hda.definitionsInFile(f)
        nodename = defin[0].nodeTypeName()
        cate = defin[0].nodeTypeCategory()

        if defin[0].isInstalled():
            b=1
            #print "'%s' already installed."%nodename

        else:
            hou.hda.installFile(f)
            print "Installed '%s' of category '%s'."%(nodename, cate)

        if create:
            try:
                hda = parentnode.createNode(nodename)
                hda.setPosition(pos)
                hda.move((-.5,-.15))
            except: pass

def convertHip(file_list):
    dia = hou.ui.displayMessage('Merge only compatible with .hip files.\nConvert to .hip? (commercial license required).', buttons=('Yes','No'), severity=hou.severityType.Message, default_choice=0, close_choice=1, title='Info')
    if not dia:
        hip = promoteHip.external(file_list[0])

        if hip:
            if hip.startswith('File \''):
                msg = hip
                hip = hip.split("'")[1]
                if not hip.endswith('.hip'):
                    hou.ui.displayMessage('Failed to create .hip file.\nCommercial license not found.',buttons=('Ok',),severity=hou.severityType.Message, default_choice=0, close_choice=0, title='Info')
                    return True

                dia = hou.ui.displayMessage('%s\nMerge file?'%msg,buttons=('Yes','No'),severity=hou.severityType.Message, default_choice=0, close_choice=1, title='Info')
            
            else:
                if not hip.endswith('.hip'):
                    hou.ui.displayMessage('Failed to create .hip file.\nCommercial license not found.',buttons=('Ok',),severity=hou.severityType.Message, default_choice=0, close_choice=0, title='Info')
                    return True
                dia = False

            if not dia:
                mergeTree.run(hip)

        else:
            hou.ui.displayMessage('Failed to create hip file.',buttons=('Ok',),severity=hou.severityType.Error, default_choice=0, close_choice=0, title='Error')

def geo(file_list):
    cursor(file_list)
    randcol()
    if not parentnode: return True

    try:    pos
    except: return True
    
    sellist = []

    path = parentnode.path()
    if not path.startswith('/obj'): return True

    with hou.undos.group('Geometry Import'):
        for ind, file in enumerate(file_list):

            # Node Name
            nodename = buildName(file)

            # Create Geo Node
            if not path.startswith('/obj/') or str(parentnode.type()) == '<hou.NodeType for Object subnet>':
                continue_check = 0

                if ext == '.vdb' and Shift:
                    geo = parentnode.createNode('arnold_volume', nodename)
                    geo.parm('ar_filename').set(file)
                    geo.parm('ar_grids').set('density')
                    continue_check = 1
                else:
                    geo = parentnode.createNode('geo', nodename)

                if ind == 0:
                    geo.setPosition(pos)
                    geo.move((-.5,-.15))
                    firstpos = geo.position()
                else:
                    if ind%4 == 0:
                        geopos = ((firstpos[0],firstpos[1]-1))
                    else:
                        geopos = ((geopos[0]+3,geopos[1]))
                    geo.setPosition(geopos)

                geopos = geo.position()

                if continue_check: continue

            elif ind == 0:
                geo = parentnode

            # Node Type
            if file.lower().endswith('.abc'):
                parm = 'fileName'
                nodetype = 'alembic'
            else:
                parm = 'file'
                nodetype = 'file'

            # Create Node
            n = geo.createNode(nodetype,nodename)
            n.parm(parm).set(file)
            n.setColor(col)

            # Node Position
            if ind == 0:
                n.setPosition(pos)
                n.move((-.5,-.15))
            else:
                n.setPosition(nodepos)
                n.move((2,0))

            nodepos = n.position()
            sellist.append(n)

    #select(sellist)

def usd(file_list):
    if houVer < 18: return True

    cursor(file_list)
    randcol()

    if not parentnode: return True

    try:    pos
    except: return True
    
    sellist = []

    # Get LOP node
    if not parentnode.type().name() == 'lopnet':
        try:
            geo = parentnode.createNode('lopnet', None)
        except:
            return True

        geo.setPosition(pos)
        geo.move((-.5,-.15))
        firstpos = geo.position()

        geopos = geo.position()
    else: geo = parentnode

    with hou.undos.group('USD Import'):
        for ind, file in enumerate(file_list):

            # Node Name
            nodename = buildName(file)

            # Create Node
            n = geo.createNode('reference',nodename)
            n.parm('filepath1').set(file)
            n.setColor(col)

            # Node Position
            if ind == 0:
                n.setPosition(pos)
                n.move((-.5,-.15))
            else:
                n.setPosition(nodepos)
                n.move((2,0))

            nodepos = n.position()
            sellist.append(n)

def cop(file_list,path):
    sellist=[]
    with hou.undos.group('Image Import'):
        for ind, file in enumerate(file_list):
            node = hou.node(path).createNode('file', buildName(file))
            node.parm('filename1').set(file)
            node.setPosition(pos)
            node.shiftPosition((-.5,0))
            if ind > 0: node.shiftPosition((4*ind,0))

        sellist.append(node)

        select(sellist)

def pic(file_list):
    cursor(file_list)
    randcol()

    nodelist = ['<hou.NodeType for Vop redshift_vopnet>','<hou.NodeType for Vop arnold_materialbuilder>','<hou.NodeType for Shop redshift_vopnet>','<hou.NodeType for Shop arnold_vopnet>']
    if not parentnode: return True
    if parentnode.type().category().name() == 'CopNet':
        cop(file_list,parentnode.path())
        return
    if repr(parentnode.type()) not in nodelist: return True

    try: pos
    except: return True

    engine = parentnode.path()
    matpath = engine + '/'
    engine = hou.node(engine).type

    # ------------------
    # Material Importer
    # ------------------

    # Lists
    global pbr,anormlist,opalist,diflist,cavlist,aolist
    bumplist = ['normal','bump','nrm','bmp','nml']
    displist = ['disp','displacement','height','heightmap','dsp']
    anormlist = ['normal','nrm','nml']
    abumplist = ['bump','bmp']
    opalist = ['opacity','sprite','alpha']
    diflist = ['albedo','diffuse','col','dif']
    speclist = ['specular','spec','reflection','refl']
    rghlist = ['roughness','gloss','glss','rgh']
    refrlist = ['refraction','refr']
    emislist = ['emission','glow']
    aolist = ['ao']
    cavlist = ['cavity']
    pbr = bumplist+displist+anormlist+abumplist+opalist+diflist+speclist+rghlist+refrlist+emislist+aolist+cavlist
   
    sellist = []
    
    dirlist = file_list

    difcon = 0
    cavcon = 0
    aocon  = 0

    # Arnold
    if 'hou.ShopNode of type arnold_vopnet' in repr(engine) or '<hou.NodeType for Vop arnold_materialbuilder>' in repr(engine):
        ntypemat = 'arnold::standard_surface'
        matsurface(ntypemat,matpath)

        with hou.undos.group('Image Import'):
            for ind, file in enumerate(dirlist):
                
                ypos = 1.8
                ntype = 'arnold::image'
                nodeSetup(ind,ntype,ypos,file)
                   
                # Set Paths and Extra Options
                tex.parm('filename').set(file)
                tex.setInputGroupExpanded(None, 0)
                tex.setDetailHighFlag(1)
                sellist.append(tex)

                if not Shift:
                    # Connect AO/Cavity
                    if nodename in diflist and not difcon:
                        difcon = tex
                    elif nodename in cavlist and not cavcon:
                        cavcon = tex
                    elif nodename in aolist and not aocon:
                        aocon = tex
                   
                    # Bump Setup
                    if nodename in abumplist:
                        bump = hou.node(matpath).createNode('arnold::bump2d', None)
                        bump.setNextInput(tex)
                        bump.moveToGoodPosition(True, False, True, True)
                        sellist.append(bump)
                        # Try Connect
                        if connect:
                            try:
                                if not connect.inputs()[39]:
                                    connect.insertInput(39, bump)
                            except:
                                connect.insertInput(39, bump)
                        else:
                            for search in hou.node(matpath).children():
                                if 'hou.NodeType for Vop standard_surface' in repr(search.type()):
                                    try:
                                        parent = search.inputs()[2]
                                        if parent == None:
                                            search.insertInput(2, bump)
                                            break
                                    except:
                                        search.insertInput(2, bump)
                                        break

                    # Normal Setup
                    if nodename in anormlist:
                        bump = hou.node(matpath).createNode('arnold::bump2d', None)
                        bump.insertInput(2, tex)
                        bump.moveToGoodPosition(True, False, True, True)
                        sellist.append(bump)
                        # Try Connect
                        if connect:
                            try:
                                if not connect.inputs()[39]:
                                    connect.insertInput(39, bump)
                            except:
                                connect.insertInput(39, bump)
                        else:
                            for search in hou.node(matpath).children():
                                if 'hou.NodeType for Vop standard_surface' in repr(search.type()):
                                    try:
                                        parent = search.inputs()[2]
                                        if parent == None:
                                            search.insertInput(2, bump)
                                            break
                                    except:
                                        search.insertInput(2, bump)
                                        break
                               
                    # Displacement Setup
                    if nodename in displist:                
                        # Try Connect
                        for search in hou.node(matpath).children():
                            if 'hou.NodeType for Vop arnold_material' in repr(search.type()):
                                try:
                                    parent = search.inputs()[1]
                                    if parent == None:
                                        search.insertInput(1, tex)
                                        break
                                except:
                                    search.insertInput(1, tex)
                                    break

                    # Connect To Material
                    if connect:
                        loc = -1
                        if nodename in diflist: loc = 1
                        if nodename in speclist: loc = 5
                        if nodename in rghlist: loc = 6
                        if nodename in ['ior']: loc = 7
                        if nodename in ['metalness']: loc = 3
                        if nodename in refrlist: loc = 11
                        if nodename in emislist: loc = 48
                        if nodename in opalist: loc = 38

                        if loc != -1:
                            try:
                                if not connect.inputs()[loc]:
                                    connect.insertInput(loc, tex)
                            except:
                                connect.insertInput(loc, tex)
                               
            # Cavity AO Connect
            inval = 1
            mulfunc(inval,difcon,aocon,cavcon)

            # Set Selected
            select(sellist)

    # Redshift
    elif 'hou.VopNode of type redshift_vopnet' in repr(engine) or '<hou.ShopNode of type redshift_vopnet at /shop/redshift_vopnet1>' in repr(engine):
        ntypemat = 'redshift::Material'
        matsurface(ntypemat,matpath)

        with hou.undos.group('Image Import'):
            for ind, file in enumerate(dirlist):

                # Setup Node
                ypos = 1.5
                ntype = 'redshift::TextureSampler'
                nodeSetup(ind,ntype,ypos,file)

                # Set Paths
                tex.parm('tex0').set(file)
                sellist.append(tex)
                
                if not Shift:
                    # Connect AO/Cavity
                    if nodename in diflist and not difcon:
                        difcon = tex
                    elif nodename in cavlist and not cavcon:
                        cavcon = tex
                    elif nodename in aolist and not aocon:
                        aocon = tex

                    # Bump Setup
                    if nodename in abumplist:
                        bump = hou.node(matpath).createNode('redshift::BumpMap', None)
                        bump.setNextInput(tex)
                        bump.moveToGoodPosition(True, False, True, True)
                        sellist.append(bump)
                       
                        if connect:
                            try:
                                if not connect.inputs()[49]:
                                    connect.insertInput(49, bump)
                            except:
                                connect.insertInput(49, bump)
                        else:
                            #Try Connect
                            for search in hou.node(matpath).children():
                                if 'hou.NodeType for Vop redshift_material' in repr(search.type()):
                                    try:
                                        parent = search.inputs()[2]
                                        if parent == None:
                                            search.insertInput(2, bump)
                                            break
                                    except:
                                        search.insertInput(2, bump)
                                        break

                    # Normal Setup
                    if nodename in anormlist:
                        #Try Connect
                        tex.setInputGroupExpanded(None, 0)

                        if connect:
                            try:
                                if not connect.inputs()[49]:
                                    connect.insertInput(49, tex)
                            except:
                                connect.insertInput(49, tex)
                        else:
                            for search in hou.node(matpath).children():
                                if 'hou.NodeType for Vop redshift_material' in repr(search.type()):
                                    try:
                                        parent = search.inputs()[2]
                                        if parent == None:
                                            search.insertInput(2, tex)
                                            break
                                    except:
                                        search.insertInput(2, tex)
                                        break


                               
                    # Displacement Setup
                    if nodename in displist:
                        disp = hou.node(matpath).createNode('redshift::Displacement', None)
                        disp.setNextInput(tex)
                        disp.moveToGoodPosition(True, False, True, True)
                        sellist.append(disp)
                       
                        # Try Connect
                        for search in hou.node(matpath).children():
                            if 'hou.NodeType for Vop redshift_material' in repr(search.type()):
                                try:
                                    parent = search.inputs()[1]
                                    if parent == None:
                                        search.insertInput(1, disp)
                                        break
                                except:
                                    search.insertInput(1, disp)
                                    break

                    # Connect To Material
                    if connect:
                        loc = -1
                        if nodename in diflist: loc = 0
                        if nodename in speclist: loc = 5
                        if nodename in rghlist: loc = 7
                        if nodename in refrlist: loc = 16
                        if nodename in emislist: loc = 48
                        if nodename in opalist: loc = 47

                        if loc != -1:
                            try:
                                if not connect.inputs()[loc]:
                                    connect.insertInput(loc, tex)
                            except:
                                connect.insertInput(loc, tex)

            # Cavity AO Connect
            inval = 3
            mulfunc(inval,difcon,aocon,cavcon)

            # Set Selected
            select(sellist)

def prox(file_list,ptype):
    cursor(file_list)
    randcol()

    try: pos
    except: return True
    position = pos

    sellist = []

    path = parentnode.path()
    if not parentnode: return True
    if str(parentnode.type()) not in ['<hou.NodeType for Object subnet>','<hou.NodeType for Manager obj>']:
        return True
    
    with hou.undos.group('Geometry Import'):
        for ind, file in enumerate(file_list):

            nodename = buildName(file)

            if ptype:
                geo = hou.node(parentnode.path()).createNode('geo',nodename)
                hou.clearAllSelected()
                geo.setSelected(True,True)
                try:
                    hou.hscript('Redshift_objectSpareParameters')
                except: 
                    geo.destroy()
                    continue

            else:
                try:
                    geo = hou.node(parentnode.path()).createNode('arnold_procedural',nodename)
                except:
                    continue

            geo.setPosition(position)
            if ind == 0: 
                geo.move((-.5,-.15))
                position = geo.position()
            else: geo.move((0,-1))

            path = geo.path()
            position = geo.position()

            if ptype:
                geo.parm('RS_objprop_proxy_enable').set(1)
                geo.parm('RS_objprop_proxy_file').set(file)
                geo.createNode('redshift_proxySOP')
            else:
                geo.parm('ar_filename').set(file)

def cursor(file_list):
    global activepane,parentnode,pos
    curdesk = hou.ui.curDesktop()
    activepane = curdesk.paneTabUnderCursor()
    parentnode = ''
    if activepane:
        try:
            pos = activepane.cursorPosition()
        except:
            return True
        bounds = activepane.allVisibleRects((None,))
        over = [b[0] for b in bounds if b[1].contains(pos)]

        if over:
            parentnode = over[0]
            message = "Dropping into '%s'"%parentnode.name()
            activepane.flashMessage(parentnode.type().icon(), message, 3)

        else:
            parentnode = activepane.pwd()

    # Paste as File Parameter if Pane not found
    try:
        pos = activepane.cursorPosition()
    except:
        try:
            sel = hou.selectedNodes()[0]
            sel = hou.selectedNodes()
            check = 0
            for ind, n in enumerate(sel):
                if len(file_list) == 1: ind = 0

                for p in n.parms():
                    if 'string_type=FileReference' in repr(p.parmTemplate().type):
                        for w in p.description().lower().split(' '):
                            if w in ['map','file','tex','filename','texture','profile']:
                                p.set(file_list[ind])
                                check = 1
                                break

                        if check:   break


        except: return True

def nodeSetup(ind,ntype,ypos,file):
    global nodename,tex
    filename = file[file.rfind('/'):]

    # Node Name
    try:
        nodename = [x for x in pbr if(x in filename.lower())]
        nodename = nodename[0]
    except:
        nodename = buildName(file)

    if nodename in anormlist and ntype == 'redshift::TextureSampler':
        ntype = 'redshift::NormalMap'

    #if nodename in opalist and ntype == 'redshift::Sprite':
    #    ntype = 'redshift::NormalMap'
    
    # Create & Position
    tex = hou.node(parentnode.path()).createNode(ntype, nodename)
    if ind == 0:
        tex.setPosition(pos)
    else:
        cpos = hou.selectedNodes()[0].position()
        cpos = (cpos[0], cpos[1]-ypos)
        tex.setPosition(cpos)
    tex.setSelected(True, True, False)
    tex.setColor(col)

def randcol():
    global col
    rand = random.uniform(0, 1)
    col = colorsys.hsv_to_rgb(rand,.5,1)
    col = hou.Color(col)

def select(sellist):
    hou.clearAllSelected()
    for node in sellist:
        node.setSelected(True, False, False)

def matsurface(ntypemat,matpath):
    # Get Material Node
    global connect
    connect = None
    for n in hou.selectedNodes():
        if n.type().name() == ntypemat:
            connect = n

    if not connect:
        for n in reversed( hou.node(matpath).children() ):
            if n.type().name() == ntypemat:
                connect = n
                break

def mulfunc(inval,difcon,aocon,cavcon):
    if difcon:
        if aocon:
            difcon.insertInput(inval, aocon)

            if cavcon:
                difcon.insertInput(inval, cavcon)

        elif cavcon:
            difcon.insertInput(inval, cavcon)

    if cavcon and aocon:
        cavcon.insertInput(inval, aocon)

def buildName(file):
    nodename = ''
    lastl = False
    for l in file[:-len(ext)].rsplit('/',1)[1]:
        if l.lower() not in list('1234567890qwertyuiopasdfghjklzxcvbnm_$'):
            if not lastl:
                nodename+='_'
            lastl = True
        else: 
            nodename+=l
            lastl = False

    # Remove _$F
    if nodename:
        if Alt:
            if len(re.findall('_\$F(?=\d+)', nodename)) > 0:
                splt = re.split('_\$F(?=\d+)', nodename)
                splt = [''.join(splt[:-1]),splt[-1]][1]
                splt = re.split(r'(^[^\D]+)', splt)[1:][0]
                nodename = nodename.replace('_$F%s'%splt,'')
            elif '_$F' in nodename:
                nodename = nodename.replace('_$F','')

        nodename = nodename.replace('$','')

    else: nodename = None

    return nodename