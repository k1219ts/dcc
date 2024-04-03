import sys
import os
import os.path
import nuke

class nukeToolbar():
    root = ''

    def create(self, title, icon):
        self.toolbar = nuke.menu('Nodes')
        return self.toolbar.addMenu(title, icon=icon)

    def setRoot(self, root):
        self.root = root

    def getGizmos(self, *args):
        if args:
            self.root = args[0]

        gizmoBuffer = []
        for items in sorted(os.walk(self.root)):
            nuke.pluginAddPath('%s'%items[0])
            for gizmo in items[2]:
                menupath = ''
                menupath = items[0].split(self.root)[-1]
                if menupath: menupath += '/'
                gizmoname = gizmo.split('.gizmo')
                if len(gizmoname)>1:
                    menuitem = menupath + gizmoname[0]
                    if menuitem[0]=='/': menuitem=menuitem[1:len(menuitem)]
                    fullpath = os.path.join(items[0], gizmo)
                    gizmoBuffer.append([menuitem, gizmoname[0], fullpath])
                    
        return gizmoBuffer
    
    def addGizmos(self, toolbar):
        gizmos = self.getGizmos(self.root)
        for each in gizmos:
            self.addMenuItem(toolbar, each[0], each[1], each[2])
                
    def getIcon(self, item, command, fullpath):
        pngfile = fullpath.split('.gizmo')[0] + '.png'
        if os.path.exists(pngfile): # if there's a png icon that's identical with the name of gizmo file, use it.
            return pngfile
        if item.find('/')>-1:
            node = item.split('/')[0]
        else:
            node = ''
        currentpng = os.path.join(self.root, node, '__default.png') # if there's a default png icon file exists in the current folder, use it.
        if os.path.exists(currentpng):
            return currentpng
        
        return os.path.join(self.root, '__default.png') # if there's no default png icon file exists in the current folder, use the one of root folder.

    def addMenuItem(self, toolbar, item, command, fullpath):
        self.toolbar = toolbar
        icon = self.getIcon(item, command, fullpath)
        self.toolbar.addCommand(item, 'nuke.createNode("%s")'%command , icon=icon)

"""
import nukeToolbar
dx_lgt = nukeToolbar.nukeToolbar()
toolbar1 = dx_lgt.create('DexterLNR', '/netapp/dexter/production/inhouse/nuke/NUKE_SCRIPT_v01/dexter/lighting/ddlnr.png')
dx_lgt.setRoot('/netapp/dexter/production/inhouse/nuke/NUKE_SCRIPT_v01/dexter/lighting')
dx_lgt.addGizmos(toolbar1)

"""
