import os
import re
import string
import glob
import pprint


class ImageParser:
    def __init__(self):
        self.ipath = ''
        self.version = ''
        self.layerData = dict() # layer data
        self.layers = list()    # layer list
        self.clayer = ''        # current layer
        self.ctxs = list()      # context list
        self.cctx = ''          # current context
        self.sublayers = list() # sub layer list
        self.csublayer = ''     # current sub layer
        self.extension = ''
        self.filename = ''

    def fileParser(self, filename):
        src = filename.split('/')
        path_index = -1
        if 'images' in src:
            path_index = src.index('images')
        if 'render' in src:
            path_index = src.index('render')

        if path_index == -1:
            return

        self.filename = filename
        self.ipath = os.path.join(*['/']+src[:path_index+1])
        self.version = src[path_index+1]

        if len(src) > (path_index + 3):
            self.clayer = src[path_index+2]

        basename = os.path.basename(filename)
        src = basename.split('.')
        if len(src) > 3:
            self.cctx = src[-3]
        else:
            self.cctx = ''

        self.csublayer = src[0].replace(self.version, '')
        # denoised image
        self.csublayer = self.csublayer.replace('_filtered', '')
        self.csublayer = self.csublayer.replace('_variance', '')

        self.extension = src[-1]


    def versionPathParser(self, vpath):
        self.ipath = os.path.dirname(vpath)
        self.version = os.path.basename(vpath)
        self.layers = self.getLayers(vpath)
        if self.layers:
            for layer in self.layers:
                img = ImageParser()
                img.layerPathParser(os.path.join(self.ipath, self.version, layer))
                self.layerData[layer] = img
        else:
            self.searchFiles()


    def layerPathParser(self, lpath):
        self.clayer = os.path.basename(lpath)
        self.version = os.path.basename(os.path.dirname(lpath))
        self.ipath = os.path.dirname(os.path.dirname(lpath))
        self.searchFiles(self.clayer)


    def parserDebug(self):
        print '# Parser Debug'
        print '\t-ipath : ', self.ipath
        print '\t-version : ', self.version
        if self.layerData:
            print '\t-layer data : ', self.layerData
        if self.layers:
            print '\t-layers : ', self.layers
        if self.clayer:
            print '\t-clayer : ', self.clayer
        if self.ctxs:
            print '\t-ctxs : ', self.ctxs
        if self.cctx:
            print '\t-cctx : ', self.cctx
        if self.sublayers:
            print '\t-sublayers : ', self.sublayers
        if self.csublayer:
            print '\t-csublayer : ', self.csublayer
        print '\t-extension : ', self.extension



    def getLayers(self, Path):
        print repr(Path)
        return os.walk(Path).next()[1]


    def searchFiles(self, Layer=None):
        dirpath = os.path.join(self.ipath, self.version)
        if Layer:
            dirpath = os.path.join(dirpath, Layer)
        source = glob.glob('{DIR}/*.*[0-9].*'.format(DIR=dirpath))
        source.sort()

        for f in source:
            basename = os.path.basename(f)
            src = basename.split('.')
            name = src[0]
            name = name.replace(self.version, '')
            name = name.replace('_filtered', '')
            name = name.replace('_variance', '')
            if not name in self.sublayers:
                self.sublayers.append(name)

        if len(self.sublayers) == 1:
            self.sublayers = list()


    def getFile(self, csublayer, context, extension):
        """
        Pre-setup variables
        : ipath, version or clayer
        """
        # Debug
        # print '# GetFile Debug'
        # print '\tcsublayer : ', csublayer
        # print '\tcontext   : ', context
        # print '\textension : ', extension

        dirpath = os.path.join(self.ipath, self.version)
        if self.clayer:
            dirpath = os.path.join(dirpath, self.clayer)

        filename = dirpath + '/'
        if csublayer:
            filename += csublayer
        filename += '*'

        if context:
            ctx = context.replace('_filtered', '')
            ctx = ctx.replace('_variance', '')
            filename += '.' + ctx + '*'

        filename += '.*[0-9].' + extension

        source = glob.glob(filename)
        source.sort()

        if not source:
            print '# Error : Not found files!'
            return None

        if not context:
            resource = list()
            for f in source:
                src = f.split('.')
                if len(src) == 3 and f.find('_variance') == -1:
                    resource.append(f)
            source = resource

        # pprint.pprint(source)
        setfilename = re.sub('\.\d+\.', '.%04d.', source[-1])
        return setfilename
