# Tactic
import requests
from tactic_client_lib import TacticServerStub
import dxConfig

# Tactic
TACTIC_IP = dxConfig.getConf("TACTIC_IP")
API_KEY = "c70181f2b648fdc2102714e8b5cb344d"
login = 'daeseok.chae'
password = 'dexter#1322'

def getShowCode(showName):
    projectName = showName
    requestParam = dict() # eqaul is requestParm = {}
    requestParam['api_key'] = API_KEY
    requestParam['name'] = projectName
    responseData = requests.get("http://{TACTIC_IP}/dexter/search/project.php".format(TACTIC_IP=TACTIC_IP), params=requestParam)

    projectInfo = responseData.json()[0]
    print projectInfo
    return projectInfo['code']


project_code = getShowCode("prat2")
shot_search_type = '%s/shot' % project_code

tactic = TacticServerStub(login=login, password=password, server=TACTIC_IP, project=project_code)

shotData = tactic.query(shot_search_type, filters=[('code', 'PS35_0180')])

print shotData

frameIn = shotData[0]['frame_in']
frameOut = shotData[0]['frame_out']

print frameOut - frameIn + 1 # 91
movCutDuration = 88012 - 87920
print movCutDuration

# # shotBuildKey = tactic.build_search_key(shot_search_type, 'LAD_5000')
# # print shotBuildKey
# # print shotData

# idList = []
# idList += range(6805612, 6805672 + 1)
# # idList += range(6805602, 6805609 + 1)
# # idList += range(6793363, 6793380 + 1)
# # idList += range(6793358, 6793361 + 1)
# # idList += [6793356]
# # idList += range(6793348, 6793350 + 1)
# # idList += range(6793344, 6793345 + 1)
# # idList += [6793342]
# # idList += [6793340]
# # idList += range(6793334, 6793338 + 1)
# # idList += range(6793317, 6793332 + 1)
# # idList += range(6793311, 6793315 + 1)
# # idList += range(6793307, 6793309 + 1)
# # idList += range(6793303, 6793305 + 1)
# # idList += range(6793299, 6793301 + 1)
# print idList
# for i in idList:
#     print i
#     tactic.undo(transaction_id=i)


# from core import excelManager
# from Define import Column2, FORMAT, TACTIC
# import datetime
#
# xlsFileName = '/prod_nas/__DD_PROD/PRAT2/edit/210224/pirates_B01_Sc_087_CG_Guide_XML_210224_all.xlsx'
# splitXlsFile = xlsFileName.split('/')
# editDate = datetime.datetime.now().strftime('%Y/%m/%d')
# if 'prod_nas' in splitXlsFile:
#     editIndex = splitXlsFile.index('edit')
#     editDate = splitXlsFile[editIndex + 1].split('_')[0]
#     print editDate
# excelMng = excelManager.ExcelMng()
# excelMng.load(xlsFileName)
#
# for i in range(1, excelMng.count('omit_list')):
#     shotName = excelMng.getRow(i, Column2.SHOT_NAME.name, 'omit_list')
#     editIssue = excelMng.getRow(i, Column2.EDIT_ISSUE.name, 'omit_list')
#     print shotName, editIssue
#     shot_search_type = '%s/shot' % project_code
#     # if velozUpdate:
#     queryData = tactic.query(shot_search_type, filters=[('code', shotName)])
#     if queryData:
#         # print queryData
#         # writeXlsRow(excelMng, row, scanListExcelMng, rowIndex, sheet='omit_list')
#         shotQueryData = queryData[0]
#         data = {}
#
#         splitEditIssueList = editIssue.split('\n')
#         note = FORMAT.EDIT_CHANGED_TITLE_NOTE.format(EDITDATE=editDate)
#
#         for splitEditIssue in splitEditIssueList:
#             print splitEditIssue
#             if 'VFX Status' in splitEditIssue:
#                 note += splitEditIssue.split(':')[-1] + '\n'
#                 data['status'] = splitEditIssue.split(':')[-1]
#             else:
#                 note += splitEditIssue
#         print note, data
#
#         shotBuildKey = tactic.build_search_key(shot_search_type, shotName)
#         try:
#             tactic.update(shotBuildKey, data)
#             # tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
#         except:
#             tactic.abort()
#         try:
#             tactic.insert('sthpw/note',
#                           {'search_type': '{projCode}/shot?project={projCode}'.format(projCode=project_code),
#                            'search_id': shotQueryData['id'],
#                            'login': TACTIC.LOGIN,
#                            'context': 'publish',
#                            'process': 'publish',
#                            'note': note})
#             # tactic, tacticTransitionCount = appendTransition(tactic, tacticTransitionCount, showName, editDate)
#         except:
#             tactic.abort()