import os, platform
import re
import json
import requests
from dxConfig import dxConfig
import DXRulebook.Interface as rb
import nuke

API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'



def getDxConfig():
    configData = None
    try:
        if 'DXCONFIGPATH' in os.environ:
            configFile = os.path.join(os.environ['DXCONFIGPATH'], 'Project.config')
            with open(configFile, 'r') as f:
                configData = json.load(f)
    except:
        pass

    return configData

def showNameToCode(show):
    showCode = None
    if 'cdh' in  show:
        show = 'cdh'

    params = {'api_key': API_KEY,
              'name': show,
              'status' : 'in_progres'}
    infos = requests.get("http://%s/dexter/search/project.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    if infos:
        showCode = infos[0]['code']

    return showCode

def getShotInfo(show, seq, shot):
    showCode = showNameToCode(show)

    params = {'api_key': API_KEY,
              'project_code': showCode,
              'sequence_code': seq,
              'q':  shot}
    infos = requests.get("http://%s/dexter/search/shot.php" % dxConfig.getConf('TACTIC_IP'), params=params).json()

    if infos:
        return infos[0]
    else:
        return None

def getTacticNote(show, shot):
    note = {}
    note['api_key'] = API_KEY
    note['project_code'] = showNameToCode(show)
    note['code'] = shot
    note['process'] = 'publish'

    infos = requests.get("http://%s/dexter/search/note.php" %(dxConfig.getConf('TACTIC_IP')), params=note).json()
    if infos:
        try:
            return infos[0]['note']
        except:
            return None
    else:
        return None

def readPlates(nukePath):
    coder = rb.Coder()
    argvTmp = coder.D.SHOW.Decode(nukePath)
    argv = coder.F.NUKE.Decode(os.path.basename(nukePath))
    argv.update(argvTmp)

    if not argv.get('seq'):
        argvTmp = coder.D.SHOT.Decode(os.path.dirname(nukePath))
        argv.update(argvTmp)

    plates = []

    plateBasePath = coder.D.PLATES.BASE.Encode(**argv)
    try:
        for desc in os.listdir(plateBasePath):
            plate = os.path.join(plateBasePath, desc)

            for ver in sorted(os.listdir(plate), reverse=True):
                plates.append(os.path.join(plate, ver))
                break
    except:
        pass

    # import paltes
    for platePath in plates:
        for file in os.listdir(platePath):
            if '.3de_bcompress' in file:
                os.remove(os.path.join(platePath, file))
                print('# deleteFile: %s' % file)

        nuke.tcl('drop', str(platePath))
        print('platePath:', platePath)

    return argv.show, argv.seq, argv.shot


def resolveOldPath():
    nodes = nuke.allNodes('Read')
    for node in nodes:
        path = node.knob('file').value()

        if not 'assetlib' in path:
            if not '_2d' in path:
                try:
                    print('-'*50)
                    print('before:', path)

                    ver = ''
                    p = re.compile('v[0-9]{3}|v[0-9]{2}')
                    for i in path.split('/'):
                        result = p.match(i)
                        if result:
                            ver = result.group()
                            if 4 > len(ver):
                                num = ver.replace('v', '').zfill(3)
                                ver = 'v'+num

                    file = os.path.basename(path)
                    coder = rb.Coder()
                    argv = coder.F.IMAGES.Decode(file)
                    argv.ver = ver

                    tmp = path.split('/')
                    show = tmp[tmp.index('show')+1]
                    if 'cdh' in show: show = 'cdh1'
                    argv.show = show

                    # print(argv)
                    if 'plates' in path:
                        if os.path.isdir(coder.D.PLATES.IMAGES.Encode(**argv)):
                            path = os.path.join(coder.D.PLATES.IMAGES.Encode(**argv), coder.F.IMAGES.BASE.Encode(**argv))
                            print('after:', path)
                    elif 'lighting' in path:
                        if os.path.isdir(coder.D.PLATES.LIGHTING.Encode(**argv)):
                            path = os.path.join(coder.D.PLATES.LIGHTING.Encode(**argv), coder.F.IMAGES.BASE.Encode(**argv))
                            print('after:', path)

                    node.knob('file').setValue(path)
                except:
                    print('%s ERROR:' % node.name(), path)
