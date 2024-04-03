#!/usr/bin/env python
'''
Miarmy Rib-Filter

LAST RELEASE
- 2017.12.18 : RMANTREE path setup change, rib compress
- 2018.05.09 : $1 (charles)- do PointsGeneralPolygons if mesh's name has "clothSim__:"
- 2018.06.13 : $2 (charles)- add exceptions (agent ids) parse args to hide objects of
- 2018.06.25 : $3 (charles)- merge crowd data
'''

import os, sys
import string
import re
import optparse
import subprocess
import glob

#from PyQt4 import QtCore, QtGui

# environment setup
application = '/netapp/backstage/pub/apps/renderman2/applications/linux'
rmantree    = glob.glob( '%s/RenderManProServer*' % application )
rmantree.sort()
# os.environ['RMANTREE'] = rmantree[-1]
os.environ['RMANTREE'] = '/netapp/backstage/pub/apps/renderman2/applications/linux/RenderManProServer-21.1'
os.environ['PYTHONPATH'] += ':%s' % os.path.join( rmantree[-1], 'bin' )
os.environ['PATH'] += ':%s' % os.path.join( rmantree[-1], 'bin' )

#print rmantree[-1]

sys.path.append( os.path.join(rmantree[-1], 'bin') )

import prman
ri = prman.Ri()

# $1 - extracting polygon namespace
EXTNS = 'clothSim___:'

# miarmy archive filter
class MiarmyArchive( prman.Rif ):
    def __init__( self, ri ):
        prman.Rif.__init__( self, ri )
        self.production = ''
        self.dorlf = 1
        self.replacesource = []
        self.identifier = ''
        self.userattributes_data = {}
        self.texture = {}
        self.procedural_data = {}
        self.objects = []
        self.block = 0
        self.extPoly = False

        self.startMotion = False
        self.motions = list()
        self.motionData = dict()
        self.polygons = list() # polygon not in motion

        # $2 : excepting ids
        self.exceptions = []
        self.hide = False

        # $3 : primpath for merging
        self.primpath = None

    #---------------------------------------------------------------------------
    # rib parse

#    def VArchiveRecord( self, type, args ):
#        if args.split()[0] != 'RLF':
#            ri.ArchiveRecord( type, args )

    def MotionBegin(self, arg):
        self.startMotion = True
        self.motions.append(arg)
        self.motionData[len(self.motions)] = list()

    def MotionEnd(self):
        self.startMotion = False

    def Transform(self, *args):
        if self.startMotion:
            self.motionData[len(self.motions)].append((ri.Transform, [args]))
        else:
            ri.Transform(args)

    def PointsGeneralPolygons( self, nloops, nvertices, vertices, params ):
        args = [nloops, nvertices, vertices, params]
        d = (ri.PointsGeneralPolygons, args)
        if self.startMotion:
            self.motionData[len(self.motions)].append(d)
        else:
            self.polygons.append(d)

    def AttributeBegin( self ):
        ri.AttributeBegin()
        self.block = 1

    def AttributeEnd( self ):
        if not self.userattributes_data:
            ri.AttributeEnd()
            return

        # $2 : if self.hide, AttributeEnd() else add attributes
        if not self.hide:
            ri.Attribute( 'identifier', {'name': self.identifier} )

            if self.replacesource and 'uniform string mapname' in self.userattributes_data.keys():
                mapname = self.userattributes_data['uniform string mapname'][0]
                mapname = mapname.replace( self.replacesource[0], self.replacesource[1] )
                self.userattributes_data['uniform string mapname'] = (mapname,)
            if self.replacesource and 'string mapname' in self.userattributes_data.keys():
                mapname = self.userattributes_data['string mapname'][0]
                mapname = mapname.replace( self.replacesource[0], self.replacesource[1] )
                self.userattributes_data['uniform string mapname'] = (mapname,)
                self.userattributes_data.pop( 'string mapname' )
            ri.Attribute( 'user', self.userattributes_data )

            if self.dorlf == 1:
                agent_name = self.identifier.split('|')[2].split('_')[-1]
                desc = 'RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,%s,' % agent_name
                ri.ArchiveRecord( 'structure', desc )

            # ri.ReverseOrientation()
            #--print("self.procedural_data : ", self.procedural_data)

            # $1 - cached polygon doesn't have Procedural
            if self.procedural_data:
                data = self.procedural_data['data']
                # dso  = data[0].split('/')[-1]
                dso  = data[0]
                opt  = data[1]
                #--print "dso : ", dso
                #--print "opt : ", opt
                ri.Procedural( (dso, opt), self.procedural_data['bound'], self.procedural_data['func'] )

            # $1 - extracting polygon when it has the namespace
            if self.extPoly:
                # $1 - motion
                for i in range(len(self.motions)):
                    ri.MotionBegin(self.motions[i])

                    for f, v in self.motionData[i+1]:
                        f(*v)

                    ri.MotionEnd()

                # $1 - polygon not in motion
                for f, v in self.polygons: f(*v)


        ri.AttributeEnd()

        self.userattributes_data.clear()
        self.procedural_data.clear()
        self.block = 0

        self.motions = list()
        self.motionData = dict()
        self.polygons = list()

        # $2 : reset self.hide
        self.hide = False


    def Attribute( self, name, params ):
        try:
            if name == 'identifier':
                if 'name' in params.keys():
                    # $1 - if name has EXTNS, set extPoly to True and remove the namespace
                    n = params['name'][0]
                    self.extPoly = EXTNS in n
                    self.identifier = n.replace(EXTNS, '') if self.extPoly else n
            elif name == 'user':
                for i in params:
                    nm = i
                    if nm.find('Body__Index') > -1:
                        self.userattributes_data['int object_id'] = (int(params[i][0]),)
                    if nm.find('Agent__Index') > -1:
                        self.userattributes_data['int group_id'] = (int(params[i][0]),)
                        self.userattributes_data['string instId'] = (str(int(params[i][0])),)

                        # $2 : if the Agent__Index is in the exceptions, self.hide = True
                        if int(params[i][0]) in self.exceptions:
                            self.hide = True

                    self.userattributes_data[nm] = params[i]
            else:
                pass
        except:
            pass


    def Surface( self, name, params ):
        # McdTxtPlastic
        if 'mapname' in params.keys():
            fn = params['mapname'][0]
            if not fn:
                return

            basepath, extension = fn.split('_diffC')
            # crowd texture
            if fn.find( '/asset/crowd/' ) > -1:
                mapname = os.path.join( os.path.dirname(os.path.dirname(basepath)), 'tex', os.path.basename(basepath) )
                self.userattributes_data['uniform string mapname'] = (mapname,)
            # asset texture
            else:
                txAssetName = basepath.split('/texture/pub/')[0]
                self.userattributes_data['uniform string txAssetName'] = (txAssetName,)
                txLayerName = os.path.basename(basepath)
                self.userattributes_data['uniform string txLayerName'] = (txLayerName,)

            # texture variation number
            index_source = re.compile(r'(_\d+|)(.\d+|).tex').findall( extension )
            if not index_source[-1] == ('', ''):
                matchRegex = index_source[-1]
                if not matchRegex[0] == "": # 0 is varName
                    self.userattributes_data['int txVarNum'] = (int(matchRegex[0][-1]),)
                if not matchRegex[1] == "": # 1 is multiUV
                    self.userattributes_data['int txmultiUV'] = (1,)


    def Procedural( self, data, bound, func ):
        # $3 - change primpath
        if self.primpath:
            _data = []
            for d in data:
                if 'MiarmySession:' in d:
                    tmp = d.split(' ')
                    tmp[1] = self.primpath
                    d = ' '.join(tmp)

                _data.append(d)
            data = tuple(_data)

        self.procedural_data['data']  = data
        self.procedural_data['bound'] = bound
        self.procedural_data['func']  = func


    def ShadingInterpolation( self, type ):
        pass

    def VArchiveRecord( self, type, args ):
        pass

    def Declare(self, name, decl):
        ri.Declare(name, decl)


# rib filtering
rif1 = MiarmyArchive( ri )
# ri.Option( 'rib', {'string asciistyle': 'indented'} )
ri.Option('rib', {
    'string format': 'askii',#'binary',
    'string asciistyle': 'indented',
    'string compression': 'gzip'
})

def rifDoIt( inputfile ):
    absinput = os.path.abspath( inputfile )
    filepath = os.path.dirname( absinput )
    dirname  = os.path.basename( filepath )
    outdir   = os.path.join( os.path.dirname(filepath), '%s_rif' % dirname )
    # make outdir
    if not os.path.exists( outdir ):
        os.makedirs( outdir )

    prman.RifInit( [rif1] )

    basename = os.path.basename( absinput )
    outfile  = os.path.join( outdir, basename )
    ri.Begin( outfile )
    prman.ParseFile( absinput )
    ri.End()
    #--print 'filtered : %s\n' % outfile

def rifUiDoIt( infile, outfile ):
    prman.RifInit( [rif1] )
    ri.Begin( outfile )
    prman.ParseFile( infile )
    ri.End()
    #--print 'filtered : %s\n' % outfile

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option( '-s', '--source', dest='source', type='string',
                          help='source file or directory' )
    optparser.add_option( '-f', '--infile', dest='infile', type='string',
                          help='source file' )
    optparser.add_option( '-o', '--outfile', dest='outfile', type='string',
                          help='out file' )
    # $2 : add argument for excepting
    optparser.add_option( '-x', '--except', dest='exceptions', type='string',
                          help='Agent IDs for exception' )
    # $3 : add argument for merging
    optparser.add_option( '-p', '--primpath', dest='primpath', type='string',
                          help='ProcPrimAsset path to merge')
    ( opts, args ) = optparser.parse_args( sys.argv )

    # $2 : get except agent ids
    if opts.exceptions:
        rif1.exceptions = [int(v) for v in opts.exceptions.split(',')]

    # $3 : get primpath
    if opts.primpath and os.path.exists(opts.primpath):
        rif1.primpath = opts.primpath

    if opts.source:
        if os.path.isfile( opts.source ):
            rifDoIt( opts.source )
        elif os.path.isdir( opts.source ):
            for i in os.listdir( opts.source ):
                if os.path.splitext(i)[-1] == '.rib':
                    rifDoIt( os.path.join(opts.source, i) )
    elif opts.infile and opts.outfile:
        rifUiDoIt( opts.infile, opts.outfile )
    else:
        print 'not support options'
