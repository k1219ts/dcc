import json
import time
import datetime
import shutil
import glob
import os.path
import sys
import subprocess
import re
import xml.dom.minidom as mx
import filecmp
import distutils.version as dv
from collections import OrderedDict
import rfm.vstruct as vstruct
from rfm.rmanAssets import *

class dxRmanAsset:

    ##
    # @brief      Constructor
    #
    # @param      self
    # @param      assetType  optional: the asset type ("nodeGraph" or "envMap"
    #                        )
    # @param      label      The asset's user-friendly label.
    #
    def __init__(self, assetType='', label='untitled'):
        '''lightweight constructor. If assetType is defined, the json skeleton
        will be constructed.'''

        self._json = {'RenderManAsset':
                      {'version': 1.0, 'label': label, 'asset': {}}}
        # file path in posix format
        self._jsonFilePath = ''
        self._version = 1.0
        self._validTypes = ['nodeGraph', 'envMap']
        self._label = label
        self._type = assetType
        self._meta = None
        self._asset = self._json['RenderManAsset']['asset']
        self._assetData = None
        self._externalFiles = []
        self._txmakeQueue = []
        self._convertToTex = False
        self._imageExts = ['.tif', '.exr', '.jpg', '.sgi', '.tga', '.iff',
                           '.dpx', '.bmp', '.hdr', '.png', '.env', '.gif',
                           '.ppm', '.xpm', '.z']
        self._textureExts = ['.tx', '.ptx']
        if assetType != '':
            self.addAsset(assetType)
            # save standard metadata: time stamp
            ts = time.time()
            tformat = '%Y-%m-%d %H:%M:%S'
            st = datetime.datetime.fromtimestamp(ts).strftime(tformat)
            self._meta['created'] = st

    ##
    # @brief      Updates references to parts of the json data structure. This
    #             is called in addAsset() and Load().
    #
    # @param      self  This object.
    #
    def __updateAliases(self):
        self._asset = self._json['RenderManAsset']['asset']
        self._assetData = self._json['RenderManAsset']['asset'][self._type]
        self._meta = self._asset[self._type]['metadata']

    ##
    # @brief      Returns the asset's type : "nodeGraph" or "envMap"
    #
    # @param      self
    #
    # @return     The asset type (string)
    #
    def type(self):
        atype = self._type
        return atype

    ##
    # @brief      Returns the asset protocol's version
    #
    # @param      self
    #
    # @return     The version (float)
    #
    def version(self):
        avers = self._version
        return avers

    ##
    # @brief      Returns the label
    #
    # @param      self
    #
    # @return     The label (string)
    #
    def label(self):
        return self._label

    ##
    # @brief      Sets the asset's label
    #
    # @param      self
    # @param      label  The new label
    #
    # @return     none
    #
    def setLabel(self, label):
        self._label = label
        self._json['RenderManAsset']['label'] = label

    def creationTime(self):
        try:
            created = self._meta['created']
        except:
            created = '----/--/-- --:--:--'
        return created

    def addMetadata(self, key, val):
        self._meta[key] = val

    def getMetadata(self, key):
        try:
            val = self._meta[key]
        except:
            val = None
        return val

    ##
    # @brief      Returns the path to the asset's json file in posix format
    #
    # @param      self
    #
    # @return     none
    #
    def jsonFilePath(self):
        return self._jsonFilePath

    def path(self):
        return os.path.dirname(self.jsonFilePath())

    ##
    # @brief      Adds an asset to the skeleton and initialises it.
    #
    # @param      self
    # @param      atype  asset type : "nodeGraph" or "envMap"
    #
    # @return     none
    #
    def addAsset(self, atype):
        '''Inserts the relevant asset structure in the dictionnary'''
        if atype in self._validTypes:
            self.type = atype
            self._asset[atype] = {'metadata': {}, 'dependencies': []}
            # add compatibility data
            cdata = {'host': {'name': None, 'version': None},
                     'renderer': {'version': None},
                     'hostNodeTypes': []}
            self._asset[atype]['compatibility'] = cdata
            if atype == 'nodeGraph':
                self._asset[atype]['nodeList'] = {}
                self._asset[atype]['connectionList'] = []
            elif atype == 'envMap':
                self._asset[atype]['specs'] = {}
            self.__updateAliases()
        else:
            raise RmanAssetError('Unknown asset type : %s' % atype)

    ##
    # @brief      Adds a connection to a connectionList
    #
    # @param      self
    # @param      src   the source node and parameter, maya-style : node.attr
    # @param      dst   the destination node and parameter, maya-style :
    #                   node.attr
    #
    # @return     none
    #
    def addConnection(self, src, dst):
        if 'connectionList' not in self._assetData.keys():
            self._assetData['connectionList'] = []
        s = src.split('.')
        d = dst.split('.')
        compound = None
        if len(d) > 2 and '[' in d[-2]:
            # if we get a multi compound like 'colorEntryList[2].color', the
            # actual renderman param name will be  'color[2]', so we do the
            # transformation here.
            compound = '.'.join(d[1:])
            index = re.search(r'\[\d+\]', d[-2]).group(0)
            d[-1] += index
        con = {'src': {'node': s[0], 'param': s[-1]},
               'dst': {'node': d[0], 'param': d[-1]}}
        if compound:
            con['dst']['compound'] = compound
        self._assetData['connectionList'].append(con)

    ##
    # @brief      Add a node to a nodeList
    #
    # @param      self      The object
    # @param      nid       node id / name
    # @param      ntype     node type
    # @param      nclass    bxdf, pattern, etc
    # @param      rmannode  The name of the corresponding RenderMan node.
    #
    # @return     none
    #
    def addNode(self, nid, ntype, nclass, rmannode, externalosl=False):
        if 'nodeList' not in self._assetData.keys():
            self._assetData['nodeList'] = OrderedDict()
        node = {'type': ntype, 'nodeClass': nclass, 'rmanNode': rmannode,
                'params': OrderedDict()}
        if externalosl:
            node['externalOSL'] = True
        # print('addNode: %s' % node)
        self._assetData['nodeList'][nid] = node

    ##
    # @brief      Add a tranform to a node in the nodeList.
    #
    # @param      self         none
    # @param      nid          the node's name
    # @param      floatValues  16 or 9 float values to specify the transform
    # @param      trStorage    storage type: default to matrix, i.e. 16 floats.
    # @param      trSpace      transform space: default to world.
    # @param      trType       transform type: default to flat, i.e. not
    #                          hierarchical.
    #
    # @return     none
    #
    def addNodeTransform(self, nid, floatValues, trNames=None,
                         trStorage=TrStorage.k_matrix,
                         trSpace=TrSpace.k_world,
                         trMode=TrMode.k_flat,
                         trType=TrType.k_coordsys):
        if floatValues is None:
            print('addNodeTransform: "%s"' % nid)
            print(str(self._json).replace(',', ',\n'))
            raise RmanAssetError('Bad float values in addNodeTransform')
        if 'transform' not in self._assetData['nodeList'][nid]:
            self._assetData['nodeList'][nid]['transforms'] = {}

        Tnode = self._assetData['nodeList'][nid]['transforms']

        # error checking
        if trSpace != TrSpace.k_world:
            raise RmanAssetError(
                'World-space only ! Other spaces not implemented yet...')
        if trMode != TrMode.k_flat:
            raise RmanAssetError(
                'Flat tranform only ! Other modes not implemented yet...')

        # store configuration
        Tnode['format'] = (trStorage, trSpace, trMode)

        if trMode == TrMode.k_flat:
            # In k_flat mode, we store all the values as a single array of
            # floats. For k_matrix, we get 16 values, for k_TRS we get 9
            # values.
            if trSpace != TrSpace.k_world:
                raise RmanAssetError('Values MUST be in world space '
                                     'for flat storage !')

            numValues = len(floatValues)
            if trStorage == TrStorage.k_matrix and numValues != 16:
                    raise RmanAssetError('Need 16 floats in matrix modes :'
                                         ' %d passed !' % numValues)
            elif trStorage == TrStorage.k_TRS and numValues != 9:
                    raise RmanAssetError('Need 9 floats in TRS modes :'
                                         ' %d passed !' % numValues)

            Tnode['values'] = floatValues

        elif trMode is TrMode.k_hierarchical:

            if trSpace is not TrSpace.k_object:
                raise RmanAssetError('Values MUST be in object space for '
                                     'hierarchical storage !')

            # In k_hierarchical mode, we store an array of tuples. Each tuple
            # decribes a transform. They are ordered bottom to top.
            # in that case, we expect floatValues to be an array of float
            # arrays and trNames to be an array of transform names.
            Tnode['values'] = []
            for name, vals in trNames, floatValues:
                Tnode['values'].append(name, vals)
        else:
            raise RmanAssetError('Unknown transform mode : %d' % trMode)

        Tnode['type'] = trType

    ##
    # @brief      Add a param to a node in a nodelist.
    #
    # @param      self   The object
    # @param      nid    node id / name
    # @param      param  name of the parameter.
    # @param      pdict  dict containing param data
    #
    # @return     none
    #
    def addParam(self, nid, param, pdict):
        # print('+ addParam %s.%s  = %s' % (nid, param, pdict))
        if param == 'mode':
            param = 'txmode'
            pdict['name'] = 'txmode'
        # print('+ addParam %s.%s  = %s' % (nid, param, pdict))
        theNode = self._assetData['nodeList'][nid]
        # we don't want to store all specs of output parameters.
        if 'output ' in pdict['type']:
            d = pdict
            unwanted = ['value', 'default']
            for k in unwanted:
                if k in d:
                    del d[k]

            theNode['params'][param] = d
            return
        # any external file path should be localized
        if pdict['type'] == 'string':
            pdict['value'] = self.processExternalFile(pdict['value'])
        # add the parameter to the list
        theNode['params'][param] = pdict

    ##
    # @brief      Save the asset as a json file
    #
    # @param      self
    # @param      filepath  Absolute path to the json file
    # @param      compact   Don't prettify the json. Defaults to False.
    # @param      progress  Progress reporting object. Use print() if None.
    #
    # @return     none
    #
    def save(self, filepath, compact=False, progress=None):
        self.registerUsedNodeTypes()
        try:
            fh = open(filepath, 'w')
        except:
            RmanAssetError('Could not create file : %s', filepath)
        if compact:
            json.dump(self._json, fh)
        else:
            json.dump(self._json, fh, sort_keys=False, indent=4,
                      separators=(',', ': '))
        fh.close()

        self._jsonFilePath = internalPath(filepath)
        self.txmake(progress=progress)
        # self.gatherExternalFiles()

    ##
    # @brief      Load a json asset file, checks its version and store the type
    #             and label
    #
    # @param      self
    # @param      filepath  The json file's absolute path in posix format
    #
    # @return     none
    #
    def load(self, filepath, localizeFilePaths=False):
        try:
            fh = open(externalPath(filepath), 'r')
        except:
            err = 'Could not open file : %s : %s' % (externalPath(filepath),
                                                     sysErr())
            raise RmanAssetError(err)
        try:
            self._json = json.load(fh)
        except:
            err = 'Failed to parse: %s : %s' % (externalPath(filepath),
                                                sysErr())
            raise RmanAssetError(err)
        fh.close()

        if float(self.version()) > 1.0:
            raise RmanAssetError('Can not read file version > %f' %
                                 self.version())

        self._type = self._json['RenderManAsset']['asset'].keys()[0]
        # print('load type: %s' % self._type)
        self._label = self._json['RenderManAsset']['label']
        self._jsonFilePath = internalPath(filepath)
        self.__updateAliases()
        if 'dependencies' in self._assetData:
            self._externalFiles = self._assetData['dependencies']
        if localizeFilePaths:
            self.localizeExternalFiles()

    ##
    # @brief      Outputs the json dict as a pretty string
    #
    # @param      self
    # @return     string
    #
    def __str__(self):
        return json.dumps(self._json, sort_keys=False, indent=4,
                          separators=(',', ': '))

    ##
    # @brief      Return the nodeList as a list of RmanAssetNode objects.
    #
    # @param      self
    #
    # @return     a list of RmanAssetNode objects.
    #
    def nodeList(self):
        if self._type != 'nodeGraph':
            raise RmanAssetError('%s asset types do not have ',
                                 'a node list !' % self._type)
        nodes = []
        for (name, data) in self._assetData['nodeList'].items():
            nodes.append(RmanAssetNode(name, data))
        return nodes

    def nodeDict(self):
        if self._type != 'nodeGraph':
            raise RmanAssetError('%s asset types do not have ',
                                 'a node list !' % self._type)
        return self._assetData['nodeList']

    ##
    # @brief      Return the connectionList as a list of
    #             RmanAssetNodeConnection objects.
    #
    # @param      self
    #
    # @return     A list of RmanAssetNodeConnection objects
    #
    def connectionList(self):
        if self._type != 'nodeGraph':
            raise RmanAssetError('%s asset types do not have a ',
                                 'connection list !' % self._type)
        clist = []
        for c in self._assetData['connectionList']:
            clist.append(RmanAssetNodeConnection(c))
        return clist

    ##
    # @brief      Create standard metadata fields : 'created', 'author',
    #             'version'. Note: this is the asset's version, not the
    #             protocol's version, which is at the top of the scope.
    #
    # @param      self
    #
    # @return     none
    #
    def stdMetadata(self):
        infos = {'created': self.creationTime()}
        try:
            infos['author'] = self._meta['author']
        except:
            infos['author'] = '-'
        try:
            infos['version'] = self._meta['version']
        except:
            infos['version'] = '1'
        return infos

    ##
    # @brief      Convert images files to textures.
    #
    # @param      self      none
    # @param      progress  Progress reporting object. Use print() if None.
    #
    # @return     none
    #
    def txmake(self, progress=None):
        assetdir = os.path.dirname(self.jsonFilePath())
        rmantree = internalPath(envGet('RMANTREE'))
        txmake = externalPath(os.path.join(rmantree, 'bin', app('txmake')))
        cmd = [txmake]
        # print('txmake for %s' % self._type)
        if self._type == 'envMap':
            cmd += ['-envlatl',
                    '-filter', 'box',
                    '-format', 'openexr',
                    '-compression', 'pxr24',
                    '-newer',
                    'src', 'dst']
        else:
            cmd += ['-resize', 'round-',
                    '-mode', 'periodic',
                    '-format', 'pixar',
                    '-compression', 'lossless',
                    '-newer',
                    'src', 'dst']
        # progress reporting
        numTextures = len(self._txmakeQueue)
        percentageDone = 0
        if progress:
            progress.Start()
        # txmake images in queue
        for img in self._txmakeQueue:
            cmd[-2] = externalPath(img)
            dirname, filename = os.path.split(img)
            cmd[-1] = externalPath(os.path.join(assetdir,
                                   os.path.splitext(filename)[0] + '.tex'))
            if progress is None:
                print('> Converting to texture :\n    %s -> %s' %
                    (cmd[-2], cmd[-1]))
            else:
                progress.Update(percentageDone, msg='Converting to texture : %s'
                                % filename)
                percentageDone += (1.0/numTextures)*100
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 startupinfo=startupInfo())
            p.wait()
        if progress:
            progress.End()

    ##
    # @brief      identify and process external files like textures.
    #
    # @param      self
    # @param      stringvalue  The parameter value, potentialy a file path.
    #
    # @return     a relative filepath.
    #
    def processExternalFile(self, stringvalue):
        logExternalFiles('>> processExternalFile: %s' % stringvalue)
        if os.path.isfile(stringvalue):
            # single file
            logExternalFiles('  > it is a file')
            texturefile = stringvalue
            path, filename = os.path.split(stringvalue)
            fname, ext = os.path.splitext(filename)
            if ext.lower() in self._imageExts:
                logExternalFiles('  > this is an image and needs to be txmade')
                texturefile = fname + '.tex'
                # This is an image : add the file to our txmake queue
                if ext not in self._textureExts:
                    logExternalFiles('  > added to txmake queue')
                    self._txmakeQueue.append(stringvalue)
                else:
                    logExternalFiles('  > already in txmake queue')
                # add to the list of files we may need to copy to the
                # asset dir.
                self._externalFiles.append(texturefile)
                logExternalFiles('  > add to list of files to copy: %s'
                                 % texturefile)
                # register the path in the dependency list.
                self._assetData['dependencies'].append(texturefile)
                logExternalFiles('  > add to dependencies: %s'
                                 % texturefile)
                return texturefile
            else:
                # add to the list of files we may need to copy to the
                # asset dir.
                if ext in self._textureExts:
                    self._externalFiles.append(stringvalue)
                    logExternalFiles('  > add to list of files to copy: %s'
                                     % stringvalue)
                    # register the path in the dependency list.
                    self._assetData['dependencies'].append(filename)
                    logExternalFiles('  > add to dependencies: %s'
                                     % filename)
                    return filename
                else:
                    return str(stringvalue).encode('string_escape')
        elif '__MAPID__' in stringvalue:
                logExternalFiles('  > it is a texture atlas')
                # texture atlas : we don't txmake them for now.
                path, filename = os.path.split(stringvalue)
                # add to the list of files we may need to copy to the
                # asset dir.
                fileglob = replace(stringvalue, '__MAPID__', '*')
                self._externalFiles.append(fileglob)
                logExternalFiles('  > add to list of files to copy: %s'
                                 % fileglob)
                # register the path in the dependency list.
                self._assetData['dependencies'].append(filename)
                logExternalFiles('  > add to dependencies: %s'
                                 % filename)
                return filename
        else:
            logExternalFiles('  > not a file : only escape...')
            return str(stringvalue).encode('string_escape')

    ##
    # @brief      Copies all referenced files (textures) to the asset
    #             directory.
    #
    # @param      self
    #
    # @return     None
    #
    def gatherExternalFiles(self):
        # print('external files: %s' % self._externalFiles)
        if len(self._externalFiles):
            root = os.path.split(self._jsonFilePath)[0]
            for dep in self._externalFiles:
                # print(dep)
                src = []
                if os.path.isfile(dep):
                    src.append(dep)
                elif '*' in dep:
                    src = glob.glob(dep)
                for s in src:
                    srcp = s
                    dstp = os.path.join(root, os.path.basename(s))
                    print('> copy external file : %s -> %s' % (srcp, dstp))
                    try:
                        shutil.copy(externalPath(srcp), externalPath(dstp))
                    except:
                        # is the file already there ?
                        this_is_wrong = True
                        if os.path.exists(dstp):
                            # is it the exact same file ?
                            if filecmp.cmp(externalPath(dstp),
                                           externalPath(srcp)):
                                # this is the same file : nothing to do
                                this_is_wrong = False
                        if this_is_wrong:
                            print('Could not copy: %s to %s' % (srcp, dstp))
                            print('>> Unexpected error:', sys.exc_info()[0])
                            raise
        else:
            print('> no external file to copy')

    ##
    # @brief      modifies the texture paths to point to the asset directory
    #
    # @param      self
    #
    # @return     None
    #
    def localizeExternalFiles(self):
        if 'nodeList' not in self._assetData:
            return

        root = os.path.dirname(self._jsonFilePath)

        for (nk, nv) in self._assetData['nodeList'].items():
            # print('nk=%s' % nk)
            for (pk, pv) in nv['params'].items():
                # print('+ pk=%s pv=%s' % (pk,pv))
                Pnode = self._assetData['nodeList'][nk]['params'][pk]
                if 'string' not in pv['type']:
                    continue
                if '[' in pv['type']:
                    for i in range(len(pv['value'])):
                        if pv['value'][i] not in self._externalFiles:
                            continue
                        Pnode['value'][i] = os.path.join(root, pv['value'][i])
                else:
                    if pv['value'] in self._externalFiles:
                        Pnode['value'] = os.path.join(root, pv['value'])

    ##
    # @brief      Gets the path to a dependency file (*.tex, *.oso, etc).
    #
    # @param      self           The object
    # @param      shortFileName  The short file name, i.e. 'diffmap.tex'
    #
    # @return     The fully qualified dependency path.
    #
    def getDependencyPath(self, shortFileName):
        root = os.path.dirname(self._jsonFilePath)
        depfile = os.path.join(root, shortFileName)
        # print('depfile = %s' % depfile)
        if os.path.exists(depfile):
            # print('depfile exists')
            return depfile
        else:
            # print('depfile missing')
            return None

    ##
    # @brief      topological sort of our shading graph
    #
    # @param      self            none
    # @param      graph_unsorted  A dict describing all edges of each node. It
    #                             should look like this:
    #                             {0:[8,2], 1:[12,15], 2:[6], 3:[12,11], ...}
    #
    # @return     A sorted version of the input graph.
    #
    def topo_sort(self, graph_unsorted):
        graph_sorted = []
        graph_unsorted = dict(graph_unsorted)

        while graph_unsorted:
            acyclic = False
            for node, edges in graph_unsorted.items():
                for edge in edges:
                    if edge in graph_unsorted:
                        break
                else:
                    acyclic = True
                    del graph_unsorted[node]
                    graph_sorted.append((node, edges))
            if not acyclic:
                raise RmanAssetError('Cyclic dependency detected !')

        return graph_sorted

    ##
    # @brief      Output RIB for this asset. First, we build an unordered graph
    #             and sort it, to be able to output the RIB statements in the
    #             right order.
    #
    # @param      self  this object
    #
    # @return     a RIB string
    #
    def getRIB(self):
        nodes = self.nodeList()
        conns = self.connectionList()

        # build a nodename to idx dict
        namedict = {}
        for i in range(len(nodes)):
            namedict[nodes[i].name()] = i

        # build a dependency list / unordered graph
        deps = {}
        for i in range(len(nodes)):
            deps[i] = []
            for c in conns:
                if c.dstNode() == nodes[i].name():
                    deps[i].append(namedict[c.srcNode()])
        # print(deps)

        # sort the graph, leaf to root
        sortedgraph = self.topo_sort(deps)
        # print(sortedgraph)

        # pass a node dict to simplify vstruct lookups
        nodeDict = {}
        for n in nodes:
            nodeDict[n.name()] = n

        rib = ''
        for g in sortedgraph:
            #  build a paramname:'srcnode:srcparam' dict for connections
            cnx = {}
            thisnode = nodes[g[0]]
            for i in g[1]:
                for c in conns:
                    if c.dstNode() == thisnode.name():
                        # make sure the node handle doesn't contain a ':' !
                        cnx[c.dstParam()] = '%s:%s' % (c.srcNodeHandle(),
                                                       c.srcParam())
            # print('\ncnx: %s -> %s' % (thisnode.name(), cnx))
            rib += thisnode.getRIB(cnx, nodeDict)
        # print('-'*70)
        # print(rib)
        # print('-'*70)
        return rib

    ##
    # @brief      Gather infos from an image's header. We use sho for now, but
    #             hopefuly we will switch to OpenImage IO in the future.
    #
    # @param      self  this object
    # @param      img   Full path to the image file
    #
    # @return     The spec dictionnary stored in the json file.
    #
    def getTextureHeader(self, img):
        specs = {}
        if os.path.exists(img):
            rmantree = internalPath(envGet('RMANTREE'))
            sho = os.path.join(rmantree, 'bin', app('sho'))
            cmd = [externalPath(sho), '-info', externalPath(img)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 startupinfo=startupInfo(),
                                 bufsize=-1)
            out, err = p.communicate()
            # print(out)
            # print(err)
            toks = err.split('\n')
            for tok in toks:
                tok = re.sub('\s{2,50}', '\t', tok)
                kv = tok.split('\t')
                if len(kv) < 2:
                    continue
                try:
                    specs[kv[0]] = eval(kv[1])
                except:
                    if '(' in kv[1]:
                        kv[1] = re.sub('\s', ',', kv[1])
                    try:
                        specs[kv[0]] = eval(kv[1])
                    except:
                        specs[kv[0]] = kv[1]
        else:
            err = 'Invalid image path : %s' % img
            raise RmanAssetError(err)
        return specs

    ##
    # @brief      Build the spec dict and stores it in the json struct.
    #
    # @param      self  this object
    # @param      img   full path to the image file.
    #
    def addTextureInfos(self, img):
        # which format is it ?
        fspecs = self.getTextureHeader(img)
        specFields = {'Original File Format': 'originalFormat',
                      'Original Size': 'originalSize',
                      'Original Bits Per Sample': 'originalBitDepth',
                      'Image Description': 'description',
                      'Display Window Origin': 'displayWindowOrigin',
                      'Display Window Size': 'displayWindowSize'}
        for k, v in specFields.iteritems():
            try:
                val = fspecs[k]
            except:
                raise RmanAssetError('Could not read %s !' % k)
            self._assetData["specs"][v] = val
        tex = self.processExternalFile(img)
        self._assetData["specs"]['filename'] = tex

    ##
    # @brief      Returns the name of the environment map
    #
    # @param      self  this object
    #
    # @return     the name as a string
    #
    def envMapName(self):
        if self._type != 'envMap':
            raise RmanAssetError('This is not an envMap asset !')
        name = self._assetData["specs"]['filename']
        # print('envMapName: %s' % name)
        return name

    ##
    # @brief      Returns the full path to the assets's environment texture
    #             file.
    #
    # @param      self  this object
    #
    # @return     the path as a string
    #
    def envMapPath(self):
        if self._type != 'envMap':
            print(self)
            raise RmanAssetError('%s is not an envMap asset !' % self._label)
        fpath = os.path.dirname(self._jsonFilePath)
        # print('envMapPath: %s' % fpath)
        fpath = os.path.join(fpath, self.envMapName())
        # print('envMapPath: %s' % fpath)
        return fpath

    ##
    # @brief      Allows an asset exporter to register host-specific node
    #             types. They will be saved in the json file so another
    #             importer may decide if they can safely rebuild this asset.
    #
    # @param      self      this object
    # @param      nodetype  The node type to register. It will only be added if
    #                       not already in the list.
    #
    def registerHostNode(self, nodetype):
        cdata = self._assetData['compatibility']
        if nodetype not in cdata['hostNodeTypes']:
            cdata['hostNodeTypes'].append(nodetype)

    ##
    # @brief      Set the values of the compatibility dict
    #
    # @param      self             this object
    # @param      hostName         The application in which that asset was
    #                              created (Maya, Katana, Houdini, Blender,
    #                              etc)
    # @param      hostVersion      The host app's version string. The version
    #                              string should be compatible with python's
    #                              distutils.version module for comparison
    #                              purposes. Should contain at least one dot.
    #                              If not, add '.0' to your version string.
    # @param      rendererVersion  The current version of the renderer. Should
    #                              contain at least one dot. If not, add '.0'
    #                              to your version string.
    #
    def setCompatibility(self, hostName=None, hostVersion=None,
                         rendererVersion=None):
        cdata = self._assetData['compatibility']
        if hostName is not None:
            cdata['host']['name'] = hostName
        if hostVersion is not None:
            cdata['host']['version'] = hostVersion
        if rendererVersion is not None:
            cdata['renderer']['version'] = rendererVersion

    ##
    # @brief      Called by importers to check if this asset can be safely
    #             imported. Potential incompatibilities will trigger a message.
    #             We only return False if the asset contains host-specific
    #             nodes for which we have no replacement. To support foreign
    #             host-specific nodes, an importer can implement an equivalent
    #             node with the same name and inputs/outputs and make sure they
    #             appear in the validNodeTypes list.
    #
    # @param      self             this object
    # @param      hostName         The importer's host
    # @param      hostVersion      The importer's host version. Should contain
    #                              at least one dot. If not, add '.0' to your
    #                              version string.
    # @param      rendererVersion  The current renderer version. Should contain
    #                              at least one dot. If not, add '.0' to your
    #                              version string.
    # @param      validNodeTypes   A list of node types the importer can safely
    #                              handle.
    #
    # @return     True if compatible, False otherwise.
    #
    def IsCompatible(self, hostName=None, hostVersion=None,
                     rendererVersion=None, validNodeTypes=[]):
        try:
            cdata = self._assetData['compatibility']
        except:
            # if the compatibility data is missing, we are dealing with an
            # old file.
            print('Warning: compatibility data is missing')
            return True

        sameHostOK = False
        if hostName is not None:
            if cdata['host']['name'] == hostName:
                sameHostOK = True

        if hostVersion is not None:
            try:
                assetVersion = dv.StrictVersion(cdata['host']['version'])
                thisVersion = dv.StrictVersion(hostVersion)
            except:
                assetVersion = dv.LooseVersion(cdata['host']['version'])
                thisVersion = dv.LooseVersion(hostVersion)
            if assetVersion <= thisVersion:
                pass
            else:
                if len(cdata['hostNodeTypes']) > 0:
                    print ('This asset contains %s %s nodes and may not '
                           'be compatible' % (cdata['host']['name'],
                                              cdata['host']['version']))

        if rendererVersion is not None:
            try:
                assetVersion = dv.StrictVersion(cdata['renderer']['version'])
                thisVersion = dv.StrictVersion(rendererVersion)
            except:
                assetVersion = dv.LooseVersion(cdata['renderer']['version'])
                thisVersion = dv.LooseVersion(rendererVersion)
            if assetVersion <= thisVersion:
                pass
            else:
                print ('This asset was created for RenderMan %s and may not '
                       'be compatible' % (cdata['renderer']['version']))

        if not sameHostOK:
            # Are there any host-specific nodes in that asset ?
            # If not, we consider this asset compatible.
            if len(cdata['hostNodeTypes']) > 0:
                # if the validNodeTypes list has been passed, check if we have
                # all required nodes...
                allHostNodesAreAvailable = False
                if len(validNodeTypes):
                    allHostNodesAreAvailable = True
                    for n in cdata['hostNodeTypes']:
                        if n not in validNodeTypes:
                            allHostNodesAreAvailable = False
                            break
                if not allHostNodesAreAvailable:
                    print ('This asset was created with %s %s and is not '
                           'compatible' % (cdata['renderer']['version']))
                    print('Missing node types : %s' % cdata['hostNodeTypes'])
                return False

        return True

    def registerUsedNodeTypes(self):
        nodetypes = []
        if self._type is 'nodeGraph':
            for k, v in self._assetData['nodeList'].iteritems():
                nodetypes.append(v['rmanNode'])
        self._json['RenderManAsset']['usedNodeTypes'] = nodetypes

    def getUsedNodeTypes(self, asString=False):
        unt = ['no data']
        try:
            unt = self._json['RenderManAsset']['usedNodeTypes']
        except:
            # compatibility mode - to be removed soon.
            # print('%s: trying old usedNodeTypes' % self.label())
            try:
                unt = self._json['usedNodeTypes']
            except:
                # print('%s: no usedNodeTypes' % self.label())
                pass
        if asString:
            return ' '.join(unt)
        else:
            return unt
