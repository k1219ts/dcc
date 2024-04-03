import nuke
import requests
from tactic_client_lib import TacticServerStub

def getFrameRange():
    fullPath = nuke.value("root.name")
    if fullPath.startswith('/netapp/dexter/show'):
        fullPath = fullPath.replace('/netapp/dexter/show', '/show')

    # IF SCRIPT FILE IS NOT FOR PROJECT, DO NOTHING
    if not(fullPath.startswith('/show/')):
        return

    pathElement = fullPath.split('/')
    #/show/mkk/shot/FFT2/FFT2_0640/comp/script/name
    projectName = pathElement[2]
    sequenceName = pathElement[4]
    shotName = pathElement[5]
    fileName = pathElement[-1][:-3]

    #------------------------------------------------------------------------------
    # PROJECT CODE CHECK
    API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'

    params = {}
    params['api_key'] = API_KEY
    params['name'] = projectName
    prjInfos = requests.get("http://10.0.0.51/dexter/search/project.php", params=params).json()
    #------------------------------------------------------------------------------
    if prjInfos:
        projectCode = prjInfos[0]['code']
        server = TacticServerStub( login='taehyung.lee', password='dlxogud', server='10.0.0.51', project=projectCode)
        shot_exp = "@SOBJECT(%s/shot['code','%s'])" % (projectCode, shotName)
        shotInfos = server.eval(shot_exp)
        if shotInfos:
            startFrame = shotInfos[0]['frame_in']
            endFrame = shotInfos[0]['frame_out']
            return (startFrame, endFrame)
        else:
            return
    else:
        return
