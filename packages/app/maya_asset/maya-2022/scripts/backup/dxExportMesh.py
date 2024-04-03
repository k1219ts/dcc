import maya.cmds as cmds
import maya.mel as mel
import os
import re
import string
import json
import time
import getpass
import math

from pymodule.Qt import QtWidgets

import MaterialSet

#-------------------------------------------------------------------------------
#
#    Display Layer
#
#-------------------------------------------------------------------------------
def getDisplayLayer():
    result = dict()
    for i in cmds.ls(type='displayLayer'):
        if i == "defaultLayer":
            continue
        id  = cmds.getAttr( '%s.identification' % i )
        viz = cmds.getAttr( '%s.visibility' % i )
        color = cmds.getAttr( '%s.color' % i )
        order = cmds.getAttr( '%s.displayOrder' % i )
        mems  = cmds.ls( cmds.editDisplayLayerMembers(i, q=True, fn=True),
                         dag=True, type='surfaceShape', ni=True )
        if id and mems:
            result[i] = {
                    'members': mems, 'order': order, 'color': color, 'visibility': viz }
    return result

def shapeList( namelist ):
    return cmds.ls( namelist, dag=True, type='surfaceShape', ni=True )

def getTextureDisplayLayers():
    result = []
    for i in cmds.ls(type='displayLayer'):
        if i.find('_LYR') > -1 and cmds.getAttr( '%s.identification' % i ):
            result.append( i )
    return result

def getMemberShape( layer ):
    members = []
    layerMembers = cmds.editDisplayLayerMembers(layer, q=True)
    if layerMembers:
        for i in layerMembers:
            members += cmds.ls( i, dag=True, type='surfaceShape', ni=True )
    return list(set(members))

#-------------------------------------------------------------------------------
#
#    UV Layout
#
#-------------------------------------------------------------------------------
class UVLayOut:
    def __init__( self ):
        self.layers   = getTextureDisplayLayers()
        self.txlayout = None
        self.uvSets   = dict()

    # uvset base is 'map1'
    def set_uvset( self, members ):
        for i in members:
            alluvsets = cmds.polyUVSet( i, allUVSets=True, q=True )
            if len(alluvsets) > 1:
                if cmds.polyUVSet( i, currentUVSet=True, q=True )[0] != 'map1':
                    cmds.polyUVSet( i, currentUVSet=True, uvSet=alluvsets[0] )
                for u in alluvsets:
                    if u != 'map1':
                        obj = cmds.listRelatives( i, f=True, p=True )
                        if u in self.uvSets.keys():
                            self.uvSets[u].append( obj[0] )
                        else:
                            self.uvSets[u] = [ obj[0] ]

    def get_uvpos( self, members ):
        _u = []; _v = []
        for i in members:
            uvs = cmds.polyEditUV( '%s.map[:]' % i, q=True )
            if uvs:
                for x in range(0,len(uvs),2):
                    _u.append( round(uvs[x],6) )
                    _v.append( round(uvs[x+1],6) )
        _u = list(set(_u))
        _u.sort()
        _v = list(set(_v))
        _v.sort()

        u_min = _u[0]
        if u_min == int(u_min):
            u_min += 0.01
        u_max = _u[-1]
        if u_max == int(u_max):
            u_max -= 0.01
        v_min = _v[0]
        if v_min == int(v_min):
            v_min += 0.01
        v_max = _v[-1]
        if v_max == int(v_max):
            v_max -= 0.01
        return u_min, u_max, v_min, v_max

    def get_uvindex( self, members ):
        self.set_uvset( members )
        indexs = []
        uv_pos = self.get_uvpos( members )
        # min
        indexs.append( int(uv_pos[0]) + int(uv_pos[2]) * 10 )
        # max
        indexs.append( int(uv_pos[1]) + int(uv_pos[3]) * 10 )
        #
        indexs = list( set(indexs) )
        indexs.sort()
        return indexs

    def layoutInfo( self ):
        dataDict = dict()
        for layer in self.layers:
            members = getMemberShape( layer )
            if members:
                name = layer.split('_LYR')[0]
                uv_index = self.get_uvindex( members )
                dataDict[name] = dict()
                dataDict[name]['order']   = cmds.getAttr( '%s.displayOrder' % layer )
                dataDict[name]['members'] = members
                dataDict[name]['txindex'] = uv_index
        self.txlayout = dataDict
        return dataDict

    def coordinate( self, index ):
        V = int( math.floor(index/10.0) )
        U = index - V * 10
        return U, V

    def selectUV( self, members ):
        source = []
        for i in members:
            source.append( '%s.map[:]' % i )
        cmds.select( source )

    def uvposition( self, opt='init' ):
        if not self.txlayout:
            self.layoutInfo()
        if opt == 'init':
            Scale = -1
        else:
            Scale = 1
        for layer in self.txlayout:
            self.selectUV( self.txlayout[layer]['members'] )
            coordsys = self.coordinate( self.txlayout[layer]['txindex'][0] )
            cmds.polyEditUV( u=coordsys[0]*Scale, v=coordsys[1]*Scale )
        cmds.select( cl=True )

#-------------------------------------------------------------------------------
#
#    Export Mesh
#
#-------------------------------------------------------------------------------
class ExportMesh():
    def __init__( self, outputName, nodeNames ):
        self.outputName = outputName
        self.nodeNames    = nodeNames
        self.textureObjects = list()
        self.displayLayerData = dict()
        self.outputFilePath = dict()
        self.externalLogs = list()

        # plugin setup
        if not cmds.pluginInfo( 'AbcExport', l=True, q=True ):
            cmds.loadPlugin( 'AbcExport' )
        self.m_abcVer = cmds.pluginInfo( 'AbcExport', v=True, q=True )

    def clearPartition( self ):
        cmds.delete( cmds.ls(type='partition') )

    def clearDisplayLayers( self ):
        disp = cmds.ls( type='displayLayer' )
        disp.remove( 'defaultLayer' )
        cmds.delete( disp )

    def restoreDisplayLayers( self ):
        if not self.displayLayerData:
            return
        data = self.displayLayerData
        order = [''] * len( data.keys() )
        for i in data:
            order[ data[i]['order']-1 ] = i
        for i in order:
            cmds.select( shapeList(data[i]['members']) )
            layer = cmds.createDisplayLayer( name=i, nr=True )
            if data[i].has_key( 'color' ):
                cmds.setAttr( '%s.color' % layer, data[i]['color'] )
            if data[i].has_key( 'visibility' ):
                cmds.setAttr( '%s.visibility' % layer, data[i]['visibility'] )
        cmds.select( clear=True )

    def maya_export( self, filename ):
        self.clearPartition()
        # delete unused material nodes
        mel.eval( 'MLdeleteUnused' )
        if self.nodeNames:
            cmds.select( self.nodeNames )
            cmds.file( filename, f=True, op='v=0', type='mayaBinary', pr=True, es=True )
        else:
            cmds.file( filename, f=True, op='v=0', type='mayaBinary', pr=True, ea=True )

        self.outputFilePath['scene'] = [filename]
        return 'Model Maya\t: %s' % filename

    def maya_backup( self, filename ):
        backup_path = os.path.join( os.path.dirname(filename), 'backup' )
        backup_name = os.path.basename( filename ).split('_v')[0] + '.mb'
        backup = os.path.join( backup_path, backup_name )
        if not os.path.exists( backup_path ):
            os.makedirs( backup_path )
        if self.nodeNames:
            cmds.select( self.nodeNames )
            cmds.file( backup, f=True, op='v=0', type='mayaBinary', pr=True, es=True )
        else:
            cmds.file( backup, f=True, op='v=0', type='mayaBinary', pr=True, ea=True )

        self.outputFilePath['backup'] = [backup]
        return 'Model Backup\t: %s' % backup

    def alembic_tex_export( self, filename, objects ):
        logs = 'Texture Alembic\t:'

        abcOption  = '-ro -writeVisibility -attr ObjectSet -attr ObjectName '
        abcOption += '-attrPrefix rman -dataFormat ogawa -uvWrite '

        meshData = dict()
        if objects:
            for i in cmds.ls(objects, dag=True, type='surfaceShape', l=True, ni=True):
                parents = cmds.listRelatives( i, f=True, p=True )
                for p in parents:
                    if p in self.textureObjects:
                        src = p.split('|')
                        if meshData.has_key( src[1] ):
                            meshData[src[1]].append( p )
                        else:
                            meshData[src[1]] = [p]
        else:
            for i in self.textureObjects:
                src = i.split('|')
                if meshData.has_key( src[1] ):
                    meshData[src[1]].append( i )
                else:
                    meshData[src[1]] = [i]

        if not meshData:
            msg = 'Texture Alembic : ERROR - Not export texture alembic'
            return msg

        abcCmd = ''
        for g in meshData:
            cmd = ' -j "'
            for i in meshData[g]:
                cmd += '-root %s ' % i
            cmd += abcOption

            src = re.compile( r'_v\d+').findall( filename )
            if src:
                default_fn = filename.split(src[-1])[0]
            else:
                default_fn = filename.split('.abc')[0]

            if src:
                suffix = src[-1] + filename.split(src[-1])[-1]
            else:
                suffix = '.abc'

            logKey = 'tex'

            if g.find( '_mid_' ) > -1:
                fn = default_fn + '_mid' + suffix
                logKey += '_mid'
            elif g.find( '_low_' ) > -1:
                fn = default_fn + '_low' + suffix
                logKey += '_low'
            elif g.find( '_sim_' ) > -1:
                fn = default_fn + '_sim' + suffix
                logKey += '_sim'
            else:
                fn = filename

            self.outputFilePath[logKey] = [fn]

            cmd += '-f %s"' % fn
            abcCmd += cmd

            # log
            logs += '\t\t%s' % fn
        print '# result : AbcExport%s' % abcCmd
        mel.eval( 'AbcExport%s' % abcCmd )
        return logs

    def appendLogs(self, logs):
        self.externalLogs = logs

#    def Xalembic_tex_export( self, filename, objects ):
#        if not cmds.pluginInfo( 'AbcExport', l=True, q=True ):
#            cmds.loadPlugin( 'AbcExport' )
#        cmd = '-j "'
#        meshlist = list()
#        if objects:
#            for i in cmds.ls(objects, dag=True, type='surfaceShape', l=True, ni=True):
#                parents = cmds.listRelatives( i, f=True, p=True )
#                for p in parents:
#                    if p in self.textureObjects:
#                        meshlist.append( p )
#        else:
#            meshlist = list(self.textureObjects)
#        if not meshlist:
#            print '# error : Not export %s' % filename
#            return 'Texture Alembic : ERROR - Not export texture alembic'
#
#        for i in meshlist:
#            cmd += '-root %s ' % i
#        cmd += '-ro '
#        cmd += '-writeVisibility '
#        cmd += '-attr ObjectSet '
#        cmd += '-attr ObjectName '
#        cmd += '-attrPrefix rman '
#        cmd += '-dataFormat ogawa '
#        cmd += '-uvWrite '
#        cmd += '-f %s' % filename
#        cmd += '"'
#        print '# result : AbcExport %s' % cmd
#        mel.eval( 'AbcExport %s' % cmd )
#        return 'Texture Alembic\t: %s' % filename

    def alembic_export( self, filename ):
        logs = 'Model Alembic\t:'

        abcOption  = '-ro -writeVisibility -attr ObjectSet -attr ObjectName '
        abcOption += '-attrPrefix rman -dataFormat ogawa -uvWrite -writeUVSets '

        abcCmd = ''
        if self.nodeNames:
            separate_ifever = 0
            for n in self.nodeNames:
                if n.find('_mid_') > -1 or n.find('_low_') > -1 or n.find('_sim_') > -1:
                    separate_ifever += 1

            if separate_ifever > 0:
                for n in self.nodeNames:
                    cmd  = ' -j "'
                    cmd += '-root %s ' % n
                    cmd += abcOption

                    src = re.compile( r'_v\d+.abc' ).findall( filename )
                    if src:
                        default_fn = filename.split(src[-1])[0]
                    else:
                        default_fn = filename.split('.abc')[0]
                        src = ['.abc']

                    logKey = 'abc'

                    if n.find( '_mid_' ) > -1:
                        fn = default_fn + '_mid' + src[-1]
                        logKey += '_mid'
                    elif n.find( '_low_' ) > -1:
                        fn = default_fn + '_low' + src[-1]
                        logKey += '_low'
                    elif n.find( '_sim_' ) > -1:
                        fn = default_fn + '_sim' + src[-1]
                        logKey += '_sim'
                    else:
                        fn = filename

                    self.outputFilePath[logKey] = [fn]

                    cmd += '-f %s"' % fn
                    abcCmd += cmd

                    # log
                    logs += '\t\t%s' % fn

            else:
                cmd = ' -j "'
                for n in self.nodeNames:
                    cmd += '-root %s ' % n
                cmd += abcOption
                cmd += '-f %s"' % filename
                abcCmd += cmd

                self.outputFilePath['abc'] = [filename]
                # log
                logs += '\t\t%s' % filename
        else:
            cmd  = ' -j "'
            cmd += abcOption
            cmd += '-f %s"' % filename
            abcCmd += cmd

            self.outputFilePath['abc'] = [filename]

            # log
            logs += '\t\t%s' % filename

        print '# result : AbcExport%s' % abcCmd
        mel.eval( 'AbcExport%s' % abcCmd )
        return logs


#    def Xalembic_export( self, filename ):
#        if not cmds.pluginInfo( 'AbcExport', l=True, q=True ):
#            cmds.loadPlugin( 'AbcExport' )
#        cmd = '-j "'
#        for i in self.nodeNames:
#            cmd += '-root %s ' % i
#        cmd += '-ro '
#        cmd += '-writeVisibility '
#        cmd += '-attr ObjectSet '
#        cmd += '-attr ObjectName '
#        cmd += '-attrPrefix rman '
#        cmd += '-dataFormat ogawa '
#        # uv export
#        abcVer = cmds.pluginInfo( 'AbcExport', v=True, q=True )
#        if abcVer == '1.5.8':
#            cmd += '-writeUVSets '
#        else:
#            cmd += '-uvWrite '
#        cmd += '-f %s' % filename
#        cmd += '"'
#        print '# result : AbcExport %s' % cmd
#        mel.eval( 'AbcExport %s' % cmd )
#        return 'Model Alembic\t: %s' % filename

    def createHeader( self ):
        header = {}
        header['created'] = time.asctime()
        header['author']  = getpass.getuser()
        header['context'] = cmds.file( q=True, sn=True )
        return header

    def displaylayerinfo_export( self, filename ):
        data = getDisplayLayer()
        for layer in data:
            txlayer = layer.split('_LYR')[0]
            if self.uvClass.txlayout.has_key( txlayer ):
                data[layer]['txindex'] = self.uvClass.txlayout[txlayer]['txindex']
        self.displayLayerData = data
        body = {}
        body['DisplayLayer'] = data
        body['_Header'] = self.createHeader()
        # write
        jsonFile = open( filename, 'w' )
        json.dump( body, jsonFile, indent=4 )
        jsonFile.close()
        self.outputFilePath['json'] = [filename]
        return 'DisplayLayer\t: %s' % filename

    def texturelayerinfo_export( self, filename ):
        body = {}
        body['TextureLayerInfo'] = self.uvClass.txlayout
        body['_Header'] = self.createHeader()
        # write
        jsonFile = open( filename, 'w' )
        json.dump( body, jsonFile, indent=4 )
        jsonFile.close()
        self.outputFilePath['tex_json'] = [filename]
        return 'TextureLayer\t: %s' % filename

    def mesh_export( self, maya=True, abc=True, tx=True ):
        export_logs = []

        # backup
        log = self.maya_backup( self.outputName )
        export_logs.append( log )
        export_logs.append( '' )

        # init uv
        self.uvClass = UVLayOut()
        txlayout = self.uvClass.layoutInfo()
        txindexs = []
        for layer in txlayout:
            txindexs.append( txlayout[layer]['txindex'][0] )
        txindexs = list(set(txindexs))

        # texture objects
        for i in txlayout:
            for o in txlayout[i]['members']:
                self.textureObjects += cmds.listRelatives( o, f=True, p=True )

        # 2018 preset shader pipeline to need
        MaterialSet.AddMaterialSetAttribute()

        # display layer
        if maya or abc:
            file_json = '%s.json' % self.outputName
            log = self.displaylayerinfo_export( file_json )
            export_logs.append( log )
            export_logs.append( '' )

        # for texture
        if tx:
            # texture uv export
            if len(txindexs) > 1 or len(txlayout.keys()) == 1:
                file_txabc = '%s_tx.abc' % self.outputName
                file_txabc = file_txabc.replace( '/model/', '/texture/' )
                if not os.path.exists(os.path.dirname(file_txabc)):
                    os.makedirs( os.path.dirname(file_txabc) )
                # texture uv
                log = self.alembic_tex_export( file_txabc, self.nodeNames )
                export_logs.append( log )
                export_logs.append( '' )
                # texture uvSets
                if self.uvClass.uvSets:
                    for u in self.uvClass.uvSets:
                        # uvSet
                        for i in self.uvClass.uvSets[u]:
                            cmds.polyUVSet( i, currentUVSet=True, uvSet=u )
                        fn = file_txabc.replace( '_tx.abc', '_%s.abc' % u )
                        log = self.alembic_tex_export( fn, self.uvClass.uvSets[u] )
                        export_logs.append( log )
                        export_logs.append( '' )
                    # undo uvSet
                    for u in self.uvClass.uvSets:
                        for i in self.uvClass.uvSets[u]:
                            cmds.polyUVSet( i, currentUVSet=True, uvSet='map1' )
                # attributes
                file_txattr = file_txabc.replace('.abc', '.json')
                log = self.texturelayerinfo_export( file_txattr )
                export_logs.append( log )
                export_logs.append( '' )
            else:
                msg = cmds.confirmDialog( title = 'Warning - Alembic for Texture',
                                          message = 'Not support Texture UV.',
                                          messageAlign = 'center',
                                          icon = 'warning',
                                          button = ['Skip', 'Cancel'] )
                if msg == 'Cancel':
                    return

        if len(txindexs) > 1:
            self.uvClass.uvposition( opt='init' )
        #
        if abc:
            file_abc = '%s.abc' % self.outputName
            log = self.alembic_export( file_abc )
            export_logs.append( log )
            export_logs.append( '' )

        #
        if maya:
            self.clearDisplayLayers()
            file_mb = '%s.mb' % self.outputName
            log = self.maya_export( file_mb )
            export_logs.append( log )
            export_logs.append( '' )
            self.restoreDisplayLayers()

        if len(self.externalLogs) > 0:
            export_logs.append(self.externalLogs)

        # undo init uv
        if len(txindexs) > 1:
            self.uvClass.uvposition( opt='' )

        # msg
        msg = cmds.confirmDialog( title = 'Export',
                                  message = string.join( export_logs, '\n' ),
                                  messageAlign = 'center',
                                  icon = 'information',
                                  button = 'OK' )

        QtWidgets.QApplication.clipboard().setText(string.join( export_logs, '\n' ))