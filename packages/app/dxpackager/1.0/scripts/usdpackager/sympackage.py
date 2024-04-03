# coding: utf-8
import os, sys, shutil, re, glob, base64, subprocess
import dbutils

scriptsDir = os.path.dirname(__file__)

def popen(cmd):
    process = subprocess.Popen(cmd,  stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
    return process.communicate()

def symWalked(packageDir,  packFileList):
    ftpRoot = '/Disks/home'
    vndFtpId,  packageDir = packageDir.split('#')
    if packageDir[0] == '/':
        packageDir = packageDir[1:]
    packageDir = packageDir.split('/from_dexter/')[-1]
    vndFtpDir = ftpRoot+'/'+vndFtpId+'/from_dexter/'+packageDir
    logFile = vndFtpDir+'/sympkg.log'
    sshCmdB64 = 'c3NocGFzcyAtcCAucm1mZm50bHRtIyEgc3NoIC1wIDkzNDkgcm9vdEAxMC4wLjAuNjEgLW8gU3RyaWN0SG9zdEtleUNoZWNraW5nPW5v'
    lnkCmd = 'ln -svf'
    cpCmd = 'cp -vf'
    programDir = '/backstage/dcc/packages/app/dxpackager/1.0/bin'

    packCmdList = []
    packDirList = []
    sshCmd = base64.b64decode(sshCmdB64).decode('ascii')
    startLogCmd = 'echo "`date +%Y-%m-%d_%H:%M:%S` <START>" > '+logFile
    endLogCmd = 'echo "`date +%Y-%m-%d_%H:%M:%S` <END>" >> '+logFile

    mbScript = False
    for path in packFileList:
        if not  os.path.isfile(path) or '/.' in path:
            continue
        if not mbScript and '/_3d/asset/' in path and path.endswith('.mb'):
            mbScript = True

        # src = '/'+path
        src = '/'+path.split('/show/')[-1]
        vndFile = vndFtpDir+path
        vndDir = os.path.dirname(vndFile)
        vndFn = os.path.basename(vndFile)
        fileCmd = cpCmd
        if 'geom' in vndFn or vndFn.endswith('.mb') or vndFn.endswith('abc'):
            fileCmd = lnkCmd
        if not vndDir in packDirList:
            packDirList.append(vndDir)
        packCmdList.append(fileCmd+' '+src+' '+vndFile+' >> '+logFile+' 2>&1')
        dbutils.updatePackage(src, vndFile, {'pkgType': 'sym'})
    
    if mbScript:
        toVendorScriptDir = scriptsDir+'/usd2maya/toVendor'
        try: toVendorScripts = os.listdir(toVendorScriptDir)
        except: toVendorScripts = []

        vndFtpScriptDir = vndFtpDir+'/scripts'
        packCmdList.insert(0, 'mkdir -p '+vndFtpScriptDir)
        for f in toVendorScripts:
            if f.endswith('.py'):
                shutil.copy2(toVendorScriptDir+'/'+f, '/show/pipe/stuff/scripts/'+f)
                packCmdList.append(cpCmd+' '+'/show/pipe/stuff/scripts/'+f+' '+vndFtpScriptDir+'/'+f+' >> '+logFile+' 2>&1')
    
    for pd in packDirList:
        packCmdList.insert(0, 'mkdir -p '+pd)

    popen(programDir+'/'+sshCmd+' \''+startLogCmd+'\'')
    cmdCnt = 0
    partNum = 20
    while True:
        cmdPart = packCmdList[cmdCnt:cmdCnt+partNum]
        if len(cmdPart) < 1:
            break
        mlCmd = '<<MLCMD\n'+'\n'.join(cmdPart)+'\nMLCMD'
        cmdCnt += partNum
        popen(programDir+'/'+sshCmd+' '+mlCmd)
        # print programDir+'/'+sshCmd+' '+mlCmd
        
    popen(programDir+'/'+sshCmd+' \''+endLogCmd+'\'')
    popen(programDir+'/'+sshCmd+' "chown -R '+vndFtpId+':users '+vndFtpDir+'"')
