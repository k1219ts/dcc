import os
import nuke
import DXRulebook.Interface as rb
import getpass


########## stamp_info_jh ##########

def getShotName(type=None):
    path = nuke.root().name()
    coder = rb.Coder()
    argv = coder.D.WORKS.Decode(path)
    tmp = path.split(argv.departs)[-1].split('/') # ['', 'KEY', 'KEY_0010', 'precomp', 'KEY_0010_precomp_v001.nk']
    shotName = tmp[2]

    return shotName     # KSC_101_101_0010

## lighting & Fx
def getShotNameLNR(type=None):
    path = nuke.root().name()
    coder = rb.Coder()
    argv = coder.D.LIGHTING.Decode(path)
    tmpLNR = path.split(argv.pub)[-1].split('/')

    shotNameLNR = tmpLNR[3]

    return shotNameLNR     # KSC_101_101_0010

def getNkNameLNR(type=None) :
	path = nuke.root().name()
	baseName = os.path.basename(path).split('--')
	NkName = baseName[0].replace('_',' ')

	return NkName
    
def getNkName(type=None):
    path = nuke.root().name()
    baseName = os.path.basename(path)
    NkName = baseName.replace('_',' ')

    return NkName     # KSC_101_101_0010_precomp v001.nk


def getPrjName(type=None):
    path = nuke.root().name()
    prj = path.split('/')[2]
    prjName = prj.upper()

    return prjName      # KSC


def getUserName(type=None):
    worker = getpass.getuser()

    return worker      # jinhee.leec


'''
def getShotName(type=None):
    path = nuke.root().name()
    coder = rb.Coder()
    argv = coder.D.WORKS.Decode(path)
    tmp = path.split(argv.departs)[-1].split('/')
    for i in tmp:
        try:
            arg = coder.N.SHOTNAME.Decode(i)
            argv.update(arg)
        except:
            continue
    #print (argv)
    #{'departs': 'CMP', 'shot': '0010', 'seq': 'KSC_101_100', 'show': 'ksc', 'root': '/show'}

    joinPath = os.path.join(argv['seq'],argv['shot'])
    shotName = joinPath.replace('/','_')
    return str(shotName)    #KSC_101_101_0010
'''
