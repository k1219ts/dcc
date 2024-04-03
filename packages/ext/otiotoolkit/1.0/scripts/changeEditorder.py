# Tactic
from tactic_client_lib import TacticServerStub
from Define import TACTIC
import requests, os, sys
from core import calculator

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = TACTIC.API_KEY
    requestParam['name'] = projectName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC.IP), params=requestParam)

    projectInfo = responseData.json()[0]
    return projectInfo['code'], projectInfo['sync']


def main(rootDir):
    showName = calculator.parseShowName(rootDir)
    print showName
    project_code, sync = getShowCode(showName.lower())

    tactic = TacticServerStub(login=TACTIC.LOGIN, password=TACTIC.PASSWORD, server=TACTIC.IP, project=project_code)
    tactic.start()

    shot_search_type = '%s/shot' % project_code
    # rootDir = '/show/emd/screening/_closed/_canne/20210208'
    # rootDir = '/show/emd/screening/_closed/_canne/20210216'
    # rootDir = '/show/emd/screening/_closed/_canne/20210217'
    # rootDir = '/show/emd/screening/_closed/_canne/20210217_2'
    # rootDir = '/show/emd/screening/_closed/_canne/20210217_3'

    editOrderDir = os.path.join(rootDir, 'editOrder')
    if not os.path.exists(editOrderDir):
        os.makedirs(editOrderDir)

    excludeShotNameList = []
    for filename in os.listdir(rootDir):
        if not filename.startswith('.') and os.path.isfile(os.path.join(rootDir, filename)):
            print filename
            splitFileName = filename.split('.')[0].split('_')
            shotName = '%s_%s' % (splitFileName[0], splitFileName[1])
            queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
            if not queryData:
                excludeShotNameList.append(shotName)
                continue
            editOrder = queryData[0]['edit_order']

            newFileName = '%04d_%s' % (editOrder, filename)

            cmd = 'cp -rf %s %s/%s' % (os.path.join(rootDir, filename), editOrderDir, newFileName)
            print cmd
            os.system(cmd)
            # print editOrder

    print excludeShotNameList

if __name__ == '__main__':
    main(sys.argv[-1])