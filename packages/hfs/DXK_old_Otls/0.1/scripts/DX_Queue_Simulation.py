#----------------------------------------------------------------
#
# PROPRIETARY INFORMATION. This Script is proprietary to 
# DEXTER Studios Inc., and is not to be reproduced, 
# transmitted, or disclosed in any way without written permission.
#
# Produced by :
#          DEXTER Studios Inc
#                  
#              *** Kyoungha Kim *** 
#               
#----------------------------------------------------------------
# DX_TRACTOR_RENDER
#----------------------------------------------------------------
# Last Update : 2017/03/31
#
#  - Support Houdini 16.0
#
#----------------------------------------------------------------


import os, sys, subprocess, getpass
import string
import shutil
import datetime
import random
import site
import json
import requests
import hou
sys.path.append('/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages')

import tractor.api.author as author
import tractor.api.query as tq

#-------------------------- -----------------
API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
#-------------------------- -----------------

#=======================================================
# CHECK HOUDINI VERSION
#=======================================================
def getHoudiniVersion():
    version = 'hfs%s' % hou.applicationVersionString()

    return version
    
#=======================================================
# CHANGE SIMULATION TYPE ( SINGLE OR MULTI)
#=======================================================
def changeType():
    if hou.node(".").parm('type').eval() == 1:
        result = 20 # max active
    else:
        result = 1
    hou.node(".").parm('tr_maxactive').set(result)
#=======================================================
# CHANGE 'FILE NODE' PARAMETERS
#=======================================================     
def changeFileNodeErr(node):    
    for child in node.children():
        node_type = str(hou.node(child.path()).type())
        if node_type == '<hou.SopNodeType for Sop file>':
           try:
              hou.node(child.path()).parm('missingframe').set(1) # 1 is No Geomtry, 0 is Report Error
           except:
              pass
        changeFileNodeErr(child)
#-----------------------------------------------------          
def changePopnetNodeErr(node):    
    for child in node.children():
        node_type = str(hou.node(child.path()).type())
        if (node_type == '<hou.SopNodeType for Sop dopnet>') or (node_type == '<hou.NodeType for Object dopnet>'):
           try:
              hou.node(child.path()).parm('cachetodisk').set(True) # 1 is No Geomtry, 0 is Report Error
           except:
              pass
        changePopnetNodeErr(child)
        
#=======================================================
# AUTO CHANGE JOB TITLE
#=======================================================
def autoJobTitle():
    tr_sceneFileName = hou.hipFile.path().split('/')[-1].split('.')[0]
    tr_outputName = hou.node('.').parm('sopoutput').eval().split('/')[-1]

    extList = ['.bgeo', '.vdb', '.abc']
    for ext in extList:
        if ext in str(tr_outputName):
            tr_title = '[HFS] (%s) (%s)' % (tr_sceneFileName, str(tr_outputName).split(ext)[0].split('.')[0])
    
            hou.node('.').parm('tr_title').set(tr_title)
        
#-----------------------------------------------------  
def toggleCheck():
    if hou.node('.').parm('tjobtitle').eval() == 1:
        hou.node('.').parm('tr_title').set('')
    else:
        autoJobTitle()
#==========================================================================================================
#==========================================================================================================
#
#                                           +++ START SIMULATION +++
#
#==========================================================================================================
#==========================================================================================================
def main():
    #=========================================================================================================================
    # ROOT PATH & HIP PATH & HIP NAME & RENDER TYPE & ALF FILE NAME & SPOOLPATH & TRACTOR ENGINE & TRACTOR PORT & SERVICE KEY
    #=========================================================================================================================
    tmp_render_root = '/netapp/fx_cache/_simulation'

    currentTime = (str(datetime.datetime.now()).replace('-','')).replace(' ','').replace(':','').split('.')[0]
    tr_userName = getpass.getuser()

    if tr_userName == 'fx-test01':
        tr_userName = 'minhyeok.jeong'
    
    tr_hipFilePath = hou.hipFile.path() # /netapp/dexter/show/.../fx/dev/scenes/blahblah.hip
    tr_sceneFileName = tr_hipFilePath.split('/')[-1].split('.')[0] # blahblah

    tr_alfFile = tr_sceneFileName + '.alf' # blahblah.alf
    
    tr_type = hou.node(".").parm('type').eval()

    #tr_spoolPath = tmp_render_root + '/ifd/' + tr_sceneFileName + '_' + tr_userName + '_' + currentTime # /netapp/fx_cache/_render/ifd/blahblah_kyoungha.kim_currentTime
    tr_spoolPath_local = '/tmp/HFR/ifd/' + tr_alfFile.split('.')[0] + '_' + tr_userName + '_' + currentTime
    tr_tempHipPath = tmp_render_root + '/hip' # '/netapp/fx_cache/_render/hip'
    
    tractor_engine = '10.0.0.30'
    tractor_port = '80'
    
    tr_service = 'Cent7'    
    tr_taskSkey = 'Houdini'
    
    #---------------------------------------------------------------------------------------------
    
    return simSetting(tr_sceneFileName, tr_type, tr_tempHipPath, tr_hipFilePath, tr_userName, 
                    currentTime, tractor_engine, tractor_port, tr_spoolPath_local)
    
    #---------------------------------------------------------------------------------------------                    
#====================================================================================================================================================
# TRACTOR JOB RENDER SETTING
#====================================================================================================================================================   
def simSetting(tr_sceneFileName, tr_type, tr_tempHipPath, tr_hipFilePath, tr_userName, currentTime, tractor_engine, tractor_port, tr_spoolPath_local):
    
    tr_ifdlimit = 1
    
    mountNodeList = [hou.pwd().node("./ropnet/OUT")]
    
    if not mountNodeList:
       hou.ui.displayMessage('Check Node Connection. Submit is canceled..!!', 
                                buttons=('OK',), severity=hou.severityType.ImportantMessage)
       return

    else:
        #---------------------------------                
        changePopnetNodeErr(hou.node('/'))
        
        changeFileNodeErr(hou.node('/'))        
        #---------------------------------
        
        for node in mountNodeList:
            
            #=============================================
            # FRAME SETTING
            #=============================================
            
            startFrame = int(node.parm('f1').eval())
            endFrame = int(node.parm('f2').eval())
            stepFrame = int(node.parm('f3').eval())
            
            duration = endFrame - startFrame + 1
            #===============================================================================
            # OUTPUT NAME
            #===============================================================================
            
            tr_output = '%s' % node.parm('sopoutput').eval().strip()
            output_directory = tr_output.split(tr_output.split('/')[-1])[0]
            
            #=============================================
            # MAKE CACHE DIRECTORY
            #=============================================
            if not (os.path.exists(output_directory)):
                os.makedirs(output_directory, 0775)            
    
            #===============================================================================
            # CHECK STORAGE FREE SIZE
            #===============================================================================
            chkFreeSize(tr_output)    
    
            #===============================================================================
            # SET OUTPUT PATH
            #===============================================================================
             
            #==================================
            # EXT FORMAT
            #==================================
            extType = ['.bgeo', '.vdb', '.abc']
            
            for ext in extType:
                if ext in os.path.basename(tr_output):
                    
                    if isNumber((os.path.basename(tr_output).split(ext)[0]).split('.')[-1]) == True:
                
                        tr_outputName = '%s' % (os.path.basename(tr_output).replace(os.path.basename(tr_output).split(ext)[0].split('.')[-1], '$F4'))
                        
                        extFormat = tr_outputName.split('$F4')[-1]
                        
                    else:
                        tr_outputName = '%s' % (os.path.basename(tr_output))
                        
                        if '.bgeo.sc' in tr_outputName:
                            extFormat = '.bgeo.sc'
                        else:
                            extFormat = ext
                                
            #tr_outputName = os.path.basename(tr_output).replace(tr_output.split('.')[-2], '$F4')    
                
            tr_output_dir = os.path.dirname(tr_output).split(os.sep)[-1] # ex)'/dust' or '/images'
            
            # LOCAL
            tr_output_local_tmp = '/tmp/HFS/%s/%s/%s' % (tr_sceneFileName + '_' + tr_userName + '_' + currentTime, tr_output_dir, tr_outputName)    
            
            #--------------------------------------------------------------------------------
            
            tr_outputPath = os.path.dirname(tr_output)
            tr_outputPath_prev = os.sep.join(tr_outputPath.split(os.sep)[:-1])

            # LOCAL
            tr_output_localPath = os.path.dirname(tr_output_local_tmp)    
            
            #===============================================================================
            # BACK UP HIP FILE
            #===============================================================================
            
            tmpHipFile = tr_tempHipPath + '/' + tr_sceneFileName + '_' + node.name() + '_' + tr_userName + '_' + currentTime + '.hip'    
                                           
            #===============================================================================
            # SAVE HIP FILE
            #===============================================================================                       
            
            saveHipFile()         
            
            #===============================================================================
            # MAKE FOLDER FOR SPOOL, BACKUP HIP FILE
            #===============================================================================             

                        
            if not (os.path.exists(tr_tempHipPath)): # /netapp/fx_cache/_render/hip
                os.makedirs(tr_tempHipPath, 0775)                 
            
            if not (os.path.exists(tr_outputPath)):
                os.makedirs(tr_outputPath, 0775)            
                                 
            try:
                shutil.copyfile(tr_hipFilePath, tmpHipFile)
            except IOError:
                hou.ui.setStatusMessage("Copy Failed. Check Disk Space..!!",hou.severityType.Fatal)
                return            
            
            
            renderSCList = []
            for frame in range(startFrame, endFrame + 1):
                
                houdiniVersion = getHoudiniVersion()
                
                #=============================================
                # RENDER TYPE
                #=============================================
                if tr_type == 1: #multi simulation
                   sFrame = frame
                   eFrame   = frame
                else: # single simulation              
                   sFrame = frame
                   eFrame   = endFrame
               
                #=============================================
                # JOB ROOT
                #=============================================
            
                tr_job_root = hou.node(".").parm('tr_job_root').eval()
                
                #=============================================
                # ITER TITLE
                #=============================================
                iterTitle = "Frame %s" % frame
                
                #----------------------------------------------------------------------------------------------
                #----------------------------------------------------------------------------------------------
                #----------------------------------------------------------------------------------------------
                                              
                if '$F4' in tr_output_local_tmp :
                    tr_output_local = tr_output_local_tmp.split('$F4')[0]
                
                else:
                    if '.bgeo.sc' in tr_output_local_tmp:
                        tr_output_local = tr_output_local_tmp.split('.bgeo.sc')[0] + '.'
                    else:
                        tr_output_local = tr_output_local_tmp.split(extFormat)[0] + '.'
                                        
                
                #----------------------------------------------------------------------------------------------
                
                #==============================================
                # FRAME RANGE TYPE
                #==============================================
                rangeType = hou.node(".").parm('trange').eval()
                
                if rangeType == 0: # Render Any Frame
                    sFrame = int(hou.frame())
                    eFrame = int(hou.frame())
                
                #===============================================================================
                #===============================================================================
                # RENDER SCRIPT COMMAND
                #===============================================================================                 
                #===============================================================================
                
                ifdgenSC_add01 = '/netapp/backstage/pub/apps/houdini/%s/scripts/sim_script/sim_ifdgen.py %s %s %s %s' % (houdiniVersion.split('hfs')[-1][:4], tmpHipFile, node.path(), sFrame, eFrame)
                
                ifdgenSC_add02 = '%s %s %s %s' % (ifdgenSC_add01, tr_output_local, os.path.dirname(tr_hipFilePath), tr_job_root)
                                                 
                ifdgenSC_add03 = "/usr/bin/python /netapp/backstage/pub/apps/houdini/%s/scripts/sim_script/sim_script.py %s %s /tmp/HFS %s %s" % (houdiniVersion.split('hfs')[-1][:4], tr_output_localPath, tr_outputPath_prev, houdiniVersion, ifdgenSC_add02 )
                                
                
                renderSC = '%s %s %s %s %s %s %s' % (ifdgenSC_add03, tr_spoolPath_local, tr_type, tr_userName, currentTime, extFormat, rangeType)
                
                renderSCList.append(renderSC)
                                          
            #=============================================
            # METADATA (UNIQUE) & JOB COMMENT
            #=============================================            
            tr_metadata = currentTime
            tr_comment = "[Scene:%s], [Out:%s]" % (tr_hipFilePath, os.path.join(tr_outputPath, tr_outputName))
            
            #=============================================
            # MAX ACTIVE SETTING 
            #=============================================
            if tr_type == 0:
                maxActive = 1
            else:
                maxActive = hou.node('.').parm('tr_maxactive').eval()
            
            #===============================================================================
            # JOB TITLE NAME
            #===============================================================================
                        
            if hou.node('.').parm('tjobtitle').eval() == 1:
            
                if(hou.node('.').parm('tr_title').eval()):
                    tr_title = '[HFS] ' + hou.node(".").parm('tr_title').eval()
    
                else:
                    if '$F4' in str(tr_outputName):
                        tr_title = '[HFS] (%s) (%s, %s)' % (tr_sceneFileName, str(tr_outputName).split(extFormat)[0].split('.$F4')[0], extFormat[1:])
    
                    else:
                        tr_title = '[HFS] (%s) (%s, %s)' % (tr_sceneFileName, str(tr_outputName).split(extFormat)[0], extFormat[1:])
            else:

                if '$F4' in str(tr_outputName):
                    tr_title = '[HFS] (%s) (%s, %s)' % (tr_sceneFileName, str(tr_outputName).split(extFormat)[0].split('.$F4')[0], extFormat[1:])

                else:
                    tr_title = '[HFS] (%s) (%s, %s)' % (tr_sceneFileName, str(tr_outputName).split(extFormat)[0], extFormat[1:])
            
               
            #===============================================================================
            #===============================================================================
            #
            # MAKE ALF SCRIPT
            #
            #===============================================================================              
            #===============================================================================
    
            #===============================================================================
            # ALF JOB
            #===============================================================================
            
            tr_taglimit = changeMaxactive()
            
            #===============================================================================
            # SELECT SERVICE KEY
            #===============================================================================
                        
            service_cent7 = hou.node('.').parm('service_cent7').eval()
            service_vili = hou.node('.').parm('service_vili').eval()
            service_saga = hou.node('.').parm('service_saga').eval()
            service_freya = hou.node('.').parm('service_freya').eval()
            service_moon = hou.node('.').parm('service_moon').eval()
                        
            if service_cent7 == 1:
                service_key = 'Cent7'
            
            elif service_vili == 1 and service_saga == 1 and service_freya == 1 and service_moon == 1:
                service_key = 'Cent7'              
                
            else:
            
                serviceList = ['service_vili', 'service_saga', 'service_freya', 'service_moon']
                
                service_dic = {'service_vili':'Vili', 'service_saga':'Saga' , 'service_freya':'Freya', 'service_moon':'Moon'}
                
                chkServiceList = []
                for keys in serviceList:
                    if hou.node('.').parm(keys).eval() == 1:
                        chkServiceList.append(keys)
                
                serviceKeyList = []        
                for checkedKey in chkServiceList:
                    serviceKeyList.append(service_dic[checkedKey])
                    
                service_key = ' || '.join(serviceKeyList)
                
            #-------------------------------------------------------------------------       
            
            job = author.Job(title = tr_title, priority = 200, comment = tr_comment, metadata = tr_metadata, service = service_key, tags = [tr_taglimit])
            
            if tr_taglimit == 'batch':
                job.projects = [setFarm()]
            else:
                job.projects = ['fx']
            
            job.tier = tr_taglimit
            job.maxactive = maxActive
                       
            #===============================================================================
            # ALF SUB TASK
            #===============================================================================              
            subTask = author.Task(title = node.name(), serialsubtasks = 1)
                        
            #==================================================================================           
            hipPath = hou.hipFile.path() # /netapp/dexter/show/..../fx/dev/scenes/test_v010.hip
            
            strSource = hipPath.split('/')
            '''
            #===============================================================================
            # MAKE JSON SCRIPT FOR DATABASE
            #===============================================================================             
            
            if 'show' in strSource:
            
                projectName = strSource[strSource.index('show')+1] # log, ssss, mkk
                
                shotName = strSource[strSource.index('show')+4]
                
                version = strSource[-1].split('.hip')[0].split('_')[-1]
            
                if not strSource[strSource.index('show')+2] == 'asset':
                    try:
                         
                        dbFilePath = dbData(hipPath, projectName, shotName, version, startFrame, endFrame, node, currentTime, tr_output, extFormat)
                        
                        cmdArg = 'python /netapp/backstage/pub/lib/python_lib/render_to_mongo.py '
                        cmdArg += dbFilePath
                        
                        dbInsertCmd = author.Command(argv=cmdArg)
                        subTask.addCommand(dbInsertCmd)
                        
                        jsonFileDeleteCmd = author.Command(argv='rm -f %s' % dbFilePath)
                        subTask.addCommand(jsonFileDeleteCmd)        
                        
                        job.addChild(subTask)
                        
                    except:
                        print "DATABASE ERROR !!!"            
                else:
                    job.addChild(subTask)
            else:
                job.addChild(subTask)
            '''
            
            job.addChild(subTask)
            
            #===============================================================================
            # ALF SUB SUB TASK
            #===============================================================================              
            
            jobList = []
            jobInfo = ''    
            
            if tr_type == 0: # Single Simulation
            
                sub_subTask = author.Task(title = "Single Simulation " + str(startFrame) + "~" + str(endFrame), serialsubtasks = 0)
                
                sub_subTask02 = author.Task(title = 'Render Cache' + str(startFrame) + "~" + str(endFrame), serialsubtasks = 1)
                sub_subTask02.addCommand( author.Command( service = 'Houdini', tags=[ 'houdini', '%s' % houdiniVersion ], argv=[ renderSCList[0] ] ) )
            
                sub_subTask.addChild( sub_subTask02 )
                subTask.addChild( sub_subTask )
                
            elif tr_type == 1: # Multi Simulation
            
                sub_subTask = author.Task(title = "Multi Simulation " + str(startFrame) + "~" + str(endFrame), serialsubtasks = 0)
                
                for index in range(duration):
                    
                    currentFrame = startFrame + index
                    
                    sub_subTask02 = author.Task(title = 'Render Cache - %04d' % currentFrame, serialsubtasks = 1)
                    sub_subTask02.addCommand( author.Command( service = 'Houdini', tags=[ 'houdini', '%s' % houdiniVersion ], argv=[ renderSCList[index].replace('$F4', str(currentFrame).zfill(4)) ] ) )
                    
                    sub_subTask.addChild( sub_subTask02 )
                
                subTask.addChild( sub_subTask )
            #-------------------                
            jobList.append(job)
    
            #=============================================
            # JOB INFO
            #=============================================            
            jobInfo += '[%04d] %s\nImage : %s\n' % (mountNodeList.index(node), job.title, os.path.join(tr_output_dir, tr_outputName))
               
            author.setEngineClientParam(hostname = '10.0.0.30', port = 80, user = tr_userName, debug = False)
            
            #===============================================================================
            #===============================================================================
            #                                                                              #
            #                              +++ JOB SPOOL +++                               #
            #                                                                              #
            #===============================================================================
            #===============================================================================

            if chkSceneConv(tr_sceneFileName) == True:

                author.setEngineClientParam(hostname = '10.0.0.30', port = 80, user = tr_userName, debug = False)
                
                for aomg in jobList:
                   if tractor_engine == '10.0.0.30':
                      aomg.spool()
                
                author.closeEngineClient()                             

                hou.ui.setStatusMessage( '%s Job(s) Submitted.' % len(mountNodeList), hou.severityType.ImportantMessage)

            else:
                hou.ui.displayMessage("'%s' scene file name is incorrect." % str(tr_sceneFileName), severity=hou.severityType.ImportantMessage)   
                                    
        #-------------------------------------------------------------------------------------------------------         
        #-------------------------------------------------------------------------------------------------------                         
        #hou.ui.setStatusMessage( '%s Job(s) Submitted.' % len(mountNodeList), hou.severityType.ImportantMessage)             
        #hou.ui.displayMessage('Job(s) Submitted.', severity=hou.severityType.ImportantMessage)
        
#=========================================================================
#=========================================================================
#
#                   DATABASE JSON FILE & DATABASE JOB
#
#=========================================================================              
#=========================================================================
def dbData(hipPath, projectName, shotName, version, startFrame, endFrame, mantraNode, currentTime, tr_output, extFormat):
    
    tr_outputFile = tr_output.replace(tr_output.split(extFormat)[0].split('.')[-1], '%04d')

    # Dictionary generate
    dbdata = {}
    dbdata['platform'] = 'Houdini'
    dbdata['show'] = projectName
    dbdata['shot'] = shotName
    dbdata['process'] = 'fx'
    dbdata['context'] = 'fx'
    dbdata['artist'] = getpass.getuser()
    
    dbdata['files'] = {'render_path':[tr_outputFile],
                        'mov':None,
                        'hip_path':hipPath
                       }                                                            
    dbdata['time'] = datetime.datetime.now().isoformat()
    dbdata['version'] = version
    dbdata['is_stereo'] = False
    dbdata['is_publish'] = False
    dbdata['start_frame'] = startFrame
    dbdata['end_frame'] = endFrame
    dbdata['metadata'] = {"test":""}
    
    # Json generate
    dbctx = json.dumps(dbdata,indent=4, separators=(',',' : '))
    dbJsonFile = '/netapp/fx_cache/db_json/simulation/%s_%s_%s_%s.json' % (hipPath.split('/')[-1].split('.hip')[0], mantraNode, getpass.getuser(), currentTime)
                   
    f = open( dbJsonFile, 'w')
    f.write(dbctx)
    f.close()
    
    return dbJsonFile

#=============================================
# SAVE HIP FILE
#=============================================      

def saveHipFile():
    if hou.node(".").parm('save_hipfile').eval() == 1:
        try:
           hou.hipFile.save(file_name=None)
        except:
           hou.ui.setStatusMessage("Save Failed. Check Disk Space..!!",hou.severityType.Error)
           return        
    else:
        pass

#=============================================
# CHECK SCENE FILE CONVENTION
#=============================================          
def chkSceneConv(tr_sceneFileName):
    hipVer = str(tr_sceneFileName).split('_')[-1]
    
    if 'v'in hipVer:
        if len(hipVer.split('v')[-1]) == 3:
            if not hipVer.split('v')[-1][1] == '0':
                return True
            else:
                return False
        else:
            return False
    else:
        return False         

#=============================================
# CHECK USER TEAM
#=============================================          
        
def setFarm():
    farmType = hou.node('.').parm('setfarm').evalAsString()
    
    return farmType        
                
#=============================================
# CHECK USER TEAM
#=============================================      

def chkMemberTeam():
    #-------------------------- -----------------
    API_KEY = 'c70181f2b648fdc2102714e8b5cb344d'
    #-------------------------- -----------------
    
    fxTeamDic = {'FX 1 Team':'fx1', 'FX 2 Team':'fx2', 'FX 3 Team':'fx3', 'FX 4 Team':'fx4', 'FX 5 Team':'fx5', 'FX 6 Team':'fx6'}
    
    #FX TEAM NUMBER QUERY
    query_members = {}
    query_members['api_key'] = API_KEY
    query_members['code'] = getpass.getuser()
    
    
    memberInfos = requests.get("http://10.0.0.51/dexter/search/user.php", params=query_members).json()  

    if memberInfos['department'].split(' ')[0] == 'FX':
        return fxTeamDic[memberInfos['department']]
    else:
        return 'fx'
        
#=============================================
# CHANGE SERVICE KEY
#=============================================     
def changeService():    
    service_cent7 = hou.node('.').parm('service_cent7').eval()
    service_vili = hou.node('.').parm('service_vili').eval()
    service_saga = hou.node('.').parm('service_saga').eval()
    service_freya = hou.node('.').parm('service_freya').eval()
    service_moon = hou.node('.').parm('service_moon').eval()
    
    if service_vili == 1 or service_saga == 1 or service_freya == 1 or service_moon == 1:
        hou.node('.').parm('service_cent7').set(0)     
    
    elif service_vili == 0 and service_saga == 0 and service_freya == 0 and service_moon == 0:
        hou.node('.').parm('service_cent7').set(1)  
        
#=============================================
# CHECK NUMBER
#=============================================
def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
#=============================================
# CHANGE MAX ACTIVE
#=============================================
def changeMaxactive():
    if hou.node(".").parm('tr_maxactive').eval() == 1:
        result = 'user'
    else:
        result = 'batch'
    hou.node(".").parm('tr_taglimit').set(result)
    
    return result
    
#=============================================
# CHECK FREE SIZE
#=============================================
def chkFreeSize(outputPath):

    st = os.statvfs(outputPath.split(os.path.basename(outputPath))[0])

    freesize = st.f_bavail * st.f_frsize
    result = freesize/1024/1024/1024*0.001

    if result < 0.3: # 0.3 = 300GB
      if hou.ui.displayMessage('At least 0.1TB or more capacity is needed.\nDo u want to continue anyway?\n%s (%.1fTB Free)' 
                                % (outputPath.split('/')[-2], freesize), buttons=('OK (Enter)', 'Cancel'), severity=hou.severityType.Message, default_choice=0, close_choice=1) == 1:
         return
    else:
        pass
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------        
#-------------------------------------------------------------------------------------------------------------------------------------------------------
