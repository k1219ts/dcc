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
import json
import requests
import hou
#------------------------------------------------------------------------------ 
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

#===========================================================
# CHECK MANTRA NODE CONNECTION
#===========================================================
def nodeLoop():
    tractorNode = hou.pwd()
    nodeList = []
    
    mantraNode = tractorNode.inputs()
    for node in mantraNode:
        node_type = str(node.type())
        if (node.isBypassed()):   
           continue
        else:           
           if node_type == "<hou.NodeType for Driver ifd>":              
              nodeList.append(node)
              
    return nodeList
    
#-------------------------- 
def run():
    tmp_render_root = '/netapp/fx_cache/_render'

    currentTime = (str(datetime.datetime.now()).replace('-','')).replace(' ','').replace(':','').split('.')[0]
    tr_userName = getpass.getuser()
    
    if tr_userName == 'fx-test01':
        tr_userName = 'minhyeok.jeong'

    tr_hipFilePath = hou.hipFile.path() # /netapp/dexter/show/.../fx/dev/scenes/blahblah.hip
    tr_sceneFileName = tr_hipFilePath.split('/')[-1].split('.')[0] # blahblah

    tr_alfFile = tr_sceneFileName + '.alf' # blahblah.alf
    #tr_ifdFileName = tr_sceneFileName
    
    tr_spoolPath = tmp_render_root + '/ifd/' + tr_sceneFileName + '_' + tr_userName + '_' + currentTime # /netapp/fx_cache/_render/ifd/blahblah_kyoungha.kim_currentTime
    tr_spoolPath_local = '/tmp/HFR/ifd/' + tr_alfFile.split('.')[0] + '_' + tr_userName + '_' + currentTime
    tr_tempHipPath = tmp_render_root + '/hip' # '/netapp/fx_cache/_render/hip'
    
    tractor_engine = '10.0.0.30'
    tractor_port = '80'
    
    tr_service = 'Cent7'    
    tr_taskSkey = 'Houdini'
    
    
    return renderSetting(tr_sceneFileName, tr_spoolPath, tr_tempHipPath, tr_hipFilePath, tr_userName, 
                        currentTime, tractor_engine, tractor_port, tr_spoolPath_local)
    
#--------------------------     
def renderSetting(tr_sceneFileName, tr_spoolPath, tr_tempHipPath, tr_hipFilePath, tr_userName, currentTime, tractor_engine, tractor_port, tr_spoolPath_local):
   
    mountNodeList = nodeLoop()
    
    #if len(mountNodeList) == 0:
    if not mountNodeList:
       hou.ui.displayMessage('Check Mantra Node Connection.\nSubmit is canceled.', 
                                buttons=('OK',), severity=hou.severityType.ImportantMessage)
       return    
    
    else:
        indexNum = 0
        
        for mantraNode in mountNodeList:
            
            mantraNodeList = []
            mantraNodeList.append(mantraNode)
            
            #=============================================
            # DISABLE BACKGROUND IMAGE
            #=============================================
            
            hou.node(mantraNode.parm('camera').eval()).parm('vm_bgenable').set(False)
                        
            #=============================================
            # MANTRA OPTION SETTING
            #=============================================
            
            ifdFilePath = tr_spoolPath_local + '/' + mantraNode.name() + '/' + tr_sceneFileName + '_' + mantraNode.name() + '.$F4.ifd'
            
            
            mantraNode.parm('vm_alfprogress').set(True)
            mantraNode.parm('soho_mkpath').set(True)
            mantraNode.parm('soho_outputmode').set(True)
            mantraNode.parm('soho_diskfile').set(ifdFilePath)
            
            #=============================================
            # FRAME SETTING
            #=============================================
            startFrame = int(mantraNode.parm('f1').eval())
            endFrame = int(mantraNode.parm('f2').eval())
            stepFrame = int(mantraNode.parm('f3').eval())
            
            duration = endFrame - startFrame + 1
                        
#            print startFrame, endFrame, stepFrame, duration

            #===============================================================================
            # OUTPUT NAME
            #===============================================================================
            
            tr_output = mantraNode.parm('vm_picture').eval().strip()
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
            tr_outputName = os.path.basename(tr_output).replace(tr_output.split('.')[-2], '$F4')
            
            tr_output_dir = os.path.dirname(tr_output).split(os.sep)[-1] # ex)'/dust' or '/images'
            
            # LOCAL
            tr_output_local = '/tmp/HFR/%s/%s/%s' % (tr_spoolPath.split(os.sep)[-1], tr_output_dir, tr_outputName)

            #--------------------------------------------------------------------------------
            
            tr_outputPath = os.path.dirname(tr_output)
            tr_outputPath_prev = os.sep.join(tr_outputPath.split(os.sep)[:-1])
            
            # LOCAL
            tr_output_localPath = os.path.dirname(tr_output_local)

            #===============================================================================
            # SET DEEP PATH
            #===============================================================================
            
            tr_deep_resolver = 'None'
            tr_output_deep = 'None'
            tr_output_deep_path = 'None'        
            tr_outputName_deep = ''
            
            chk_deep = mantraNode.parm('vm_deepresolver').eval()
            
            #===============================================================================
            # CHECK DEEP RESOLVE
            #===============================================================================            
            
            if chk_deep != 'null':
            
               if chk_deep == 'shadow':
                  try:
                    tr_output_deep = mantraNode.parm('vm_dsmfilename').eval()
                    tr_deep_resolver = 'vm_dsmfilename'
                  except:
                    hou.ui.displayMessage('Job submit is terminated!!\nAdd [vm_dsmfilename] parameter.', buttons=('OK',), severity=hou.severityType.Error)                
                    return
               if chk_deep == 'camera':
                  try:
                    tr_output_deep = mantraNode.parm('vm_dcmfilename').eval()           
                    tr_deep_resolver = 'vm_dcmfilename'
                    
                  except:
                    hou.ui.displayMessage('Job submit is terminated!!\nAdd [vm_dcmfilename] parameter!!', buttons=('OK',), severity=hou.severityType.Error)
                    return                    
                
            
               tr_output_deep_path = os.path.dirname(tr_output_deep)
               
               tr_outputName_deep = os.path.basename(tr_output_deep).replace(tr_output.split('.')[-2], '$F4')
               
               # LOCAL
               tr_output_dir = os.path.dirname(tr_output_deep).split(os.sep)[-1]
               tr_output_local_deep = '/tmp/HFR/%s/%s/%s' % ( tr_spoolPath.split(os.sep)[-1], tr_output_dir, tr_outputName_deep )

            #===============================================================================
            # BACK UP HIP FILE
            #===============================================================================           
            tmpHipFile = tr_tempHipPath + '/' + tr_sceneFileName + '_' + mantraNode.name() + '_' + tr_userName + '_' + currentTime + '.hip'
            
            #===============================================================================
            # SAVE HIP FILE
            #===============================================================================                       
            
            saveHipFile()
                        
            #===============================================================================
            # MAKE FOLDER FOR SPOOL, BACKUP HIP FILE
            #===============================================================================             
            
            if not (os.path.exists(tr_spoolPath)): # ex) /netapp/fx_cache/_render/ifd/testHip_kyoungha.kim_20160810182837
                os.makedirs(tr_spoolPath, 0775)
            
            if not (os.path.exists(tr_tempHipPath)): # /netapp/fx_cache/_render/hip
                os.makedirs(tr_tempHipPath, 0775)                 
            
            if not (os.path.exists(tr_outputPath)):
                os.makedirs(tr_outputPath, 0775)            
            
            
            if tr_output_deep_path != 'None':
                if not (os.path.exists(tr_output_deep_path)):
                    os.makedirs(tr_output_deep_path, 0775)                 
            
            try:
                shutil.copyfile(tr_hipFilePath, tmpHipFile)
            except IOError:
                hou.ui.setStatusMessage("Copy Failed. Check Disk Space..!!",hou.severityType.Fatal)
                return
            
            
            if tr_output_deep == 'None':
               tr_output_local_deep = 'None'
            
            renderSCList = []
            for frame in range(startFrame, endFrame + 1):
                
                houdiniVersion = getHoudiniVersion()
            
                #=============================================
                # JOB ROOT
                #=============================================
            
                #tr_job_root = '/home/%s' % tr_userName
                tr_job_root = hou.node(".").parm('tr_job_root').eval()
                
                #=============================================
                # ITER TITLE
                #=============================================
                iterTitle = "Frame %s" % frame
                
                #===============================================================================
                # RENDER SCRIPT
                #===============================================================================                
                
                ifdgenSC_add01 = "/netapp/backstage/pub/apps/houdini/%s/scripts/render_script/render_ifdgen.py %s %s %s %s" % (houdiniVersion.split('hfs')[-1][:4], tmpHipFile, mantraNode.path(), frame, frame)
                
                ifdgenSC_add02= "%s %s %s %s %s %s" % (ifdgenSC_add01, tr_output_local.replace('$F4', str(frame).zfill(4)), 
                                                    tr_output_local_deep.replace('$F4', str(frame).zfill(4)), os.path.dirname(tr_hipFilePath), tr_job_root, tr_output_deep_path)
                
                
                ifdgenSC_add03  = "/usr/bin/python /netapp/backstage/pub/apps/houdini/%s/scripts/render_script/render_script.py %s %s /tmp/HFR %s %s" % (houdiniVersion.split('hfs')[-1][:4], tr_output_localPath, tr_outputPath_prev, houdiniVersion, ifdgenSC_add02 )
                
                #=============================================
                # RENDER COMMAND
                #=============================================
                                                
                renderSC = "%s %s/%s/%s %s %s %s" % (ifdgenSC_add03, tr_spoolPath_local, mantraNode.name(), tr_sceneFileName+"_" + mantraNode.name()+"."+str(frame).zfill(4)+".ifd", 
                                                tr_userName, currentTime, tr_deep_resolver)
                
                #print renderSC                                                                                            
                
                renderSCList.append(renderSC)
                                                
                #-------------------------------------------------------------------------------
                                                               
            tr_metadata = currentTime
            tr_comment = "[Scene:%s], [Out:%s]" % (tr_hipFilePath, os.path.join(tr_outputPath, tr_outputName))

            
            #===============================================================================
            # JOB TITLE NAME
            #===============================================================================
            
            #tr_title = '[HFR] (' + tr_sceneFileName + ') ' + mantraNode.parm('vm_picture').eval().split('/')[-1].split('.')[0] # [HFR] (blahblah_v01) renderImagesName
             
            #tr_title = '[HFR] (' + tr_sceneFileName + ') ' + str(mantraNode)            
            
            tr_title = '[HFR] (Scene: %s) (Mantra: %s)' % (tr_sceneFileName, str(mantraNode))
            
            #===============================================================================
            # MAKE ALF SCRIPT
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
                    
                
            #-----------------------------------------------------------------------------------------    
                                                      
            job = author.Job(title = tr_title, priority = 100, comment = tr_comment, metadata = tr_metadata, service = service_key, tags = [tr_taglimit])
            
            if tr_taglimit == 'batch':
                job.projects = [setFarm()]
            else:
                job.projects = ['fx']
                
            job.tier = tr_taglimit
            
            if hou.node(".").parm('tr_maxactive').eval() == 0:
                job.maxactive = 0
            
            else:
                job.maxactive = hou.node(".").parm('tr_maxactive').eval()  
            
            #===============================================================================
            # SUB TASK ( MONGO DB COMMAND ) 
            #===============================================================================              
            subTask = author.Task(title = mantraNode.name(), serialsubtasks = 1)
            
            #==================================================================================                         
            hipPath = hou.hipFile.path() # /netapp/dexter/show/..../fx/dev/scenes/test_v010.hip
            
            strSource = hipPath.split('/')
            
            #===============================
            # MAKE JSON SCRIPT FOR DATABASE
            #===============================           

            if 'show' in strSource:
                projectName = strSource[strSource.index('show')+1] # log, ssss, mkk
                
                shotName = strSource[strSource.index('show')+4]
                
                version = strSource[-1].split('.hip')[0].split('_')[-1]
                           
            
                if not strSource[strSource.index('show')+2] == 'asset':
                    try:
                        
                        dbFilePath = dbData(hipPath, projectName, shotName, version, startFrame, endFrame, mantraNode, currentTime, tr_output)
                        
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
            #===============================================================================
            # SUB SUB TASK
            #===============================================================================              
            
            sub_subTask = author.Task(title = "Render " + str(startFrame) + "~" + str(endFrame), serialsubtasks = 0)
            
            jobList = []
            jobInfo = ''
                        
            for frame in range(duration):
                                                
                currentFrame = startFrame + frame
                
                
                sub_subTaskFrame = author.Task(title="IFDgen and Mantra - %04d" % currentFrame, serialsubtasks = 1)
                

                sub_subTaskFrame.addCommand(author.Command( service = 'Houdini', tags=[ 'houdini', '%s' % houdiniVersion ], argv=[ renderSCList[frame] ] ) )
                                                                                
                sub_subTask.addChild(sub_subTaskFrame)
                
            subTask.addChild(sub_subTask)
            
            cleanUpTask= job.newTask(title="CleanUp",serialsubtasks = 1)
            cleanUpTask.newTask(title="CleanUpLayer")
            cleanUpTask.addCommand(author.Command( argv=["/bin/rm", "-rf", "%s/%s" % (tr_spoolPath, mantraNode.name())] ) )
            
            subTask.addChild(cleanUpTask)
            #-------------------
            jobList.append(job)
            #-------------------
            jobInfo += '[%04d] %s\nImage : %s\n' % (mountNodeList.index(mantraNode), job.title, os.path.join(tr_output_dir, tr_outputName))
            if chk_deep != 'null':
                jobInfo += 'Deep : %s\n%s\n' % (os.path.join(tr_output_dir, tr_outputName_deep), '-' * 80)
               
            #------------------------------------------------------------------------------------------------

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
def dbData(hipPath, projectName, shotName, version, startFrame, endFrame, mantraNode, currentTime, tr_output):

    
    tr_outputFile = tr_output.replace(tr_output.split('.')[-2], '%04d')

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
    dbdata['metadata'] = {"untitled":""}
    
    # Json generate
    dbctx = json.dumps(dbdata,indent=4, separators=(',',' : '))
    dbJsonFile = '/netapp/fx_cache/db_json/%s_%s_%s_%s.json' % (hipPath.split('/')[-1].split('.hip')[0], mantraNode, getpass.getuser(), currentTime)
                   
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

    if result < 0.1: # 0.1 = 100GB
      if hou.ui.displayMessage('At least 0.1TB or more capacity is needed.\nDo u want to continue anyway?\n%s (%.1fTB Free)' 
                                % (outputPath.split('/')[-2], freesize), buttons=('OK (Enter)', 'Cancel'), severity=hou.severityType.Message, default_choice=0, close_choice=1) == 1:
         return
    else:
        pass
        
#-------------------------------------------------------------------------------------------------------------------------------------------------------        
#-------------------------------------------------------------------------------------------------------------------------------------------------------
