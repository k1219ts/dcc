import os, sys, re, glob, time, datetime, subprocess, yaml, shutil, logging

reqDir = '/stuff/pipe/stuff/pkgreq'
reqWaitDir = reqDir+'/wait'
reqActiveDir = reqDir+'/active'
reqExpiredDir = reqDir+'/expired'
expireDay = 7

def createLogger(logName, logPath):
    if logPath == '':
        logPath = '/tmp/logs/'+logName+'.log'
    logDir = os.path.dirname(logPath)
    if not os.path.isdir(logDir):
        try: os.makedirs(logDir)
        except: pass

    if not os.path.isdir(logDir):
        print 'Cannot access log dir.', logDir
        raise

    newLogger = logging.getLogger(logName)
    newLogger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)-15s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if os.path.isdir(logDir):
        fileHandler = logging.FileHandler(logPath)
        fileHandler.setFormatter(formatter)
        newLogger.addHandler(fileHandler)

    return newLogger

sympkgLogger = createLogger('SYMPKG', '')

def waitToActive():
    try: waitFileList = os.listdir(reqWaitDir)
    except: waitFileList = []

    for wf in waitFileList:
        if not wf.endswith('.log'):
            continue

        waitFile = reqWaitDir+'/'+wf
        waitFn = os.path.basename(waitFile)
        if os.path.isfile(waitFile):
            with open(waitFile, 'r') as f:
                try: reqPkgDict = yaml.load(f, Loader=yaml.SafeLoader)
                except: reqPkgDict = None
                
                if reqPkgDict != None:
                    packageRootDir = '/vendor/ftp/packages'#os.path.expanduser('~')+'/packages'
                    for rf in reqPkgDict:
                        packagePath = os.path.normpath(packageRootDir+'/'+rf['dst'])
                        packageDir = os.path.dirname(packagePath)
                        try: os.makedirs(packageDir)
                        except: pass
                        if not os.path.isdir(packageDir):
                            sympkgLogger.error('Cannot access dir('+packageDir+').')
                            print 'Cannot access dir('+packageDir+').'
                            continue

                        if not os.path.isfile(packagePath):
                            try: os.symlink(rf['src'], packagePath)
                            except:
                                sympkgLogger.error('Symlink error('+packagePath+').')
                                print 'Symlink error('+packagePath+').'
                                continue

                            sympkgLogger.info('symlink '+rf['src']+' -> '+packagePath+' ok.')
                            print 'symlink '+rf['src']+' -> '+packagePath+' ok.'

        try:
            activeFile = reqActiveDir+'/'+waitFn
            shutil.move(waitFile, activeFile)
            now = time.time()
            atime = now
            mtime = now
            os.utime(activeFile, (atime, mtime))
            sympkgLogger.info('Wait to Active ok('+waitFile+').')
            print 'Wait to Active ok('+waitFile+').'
        except:
            sympkgLogger.error('Wait to Active error('+waitFile+').')
            print 'Wait to Active error('+waitFile+').'

def activeToExpired():
    expireDatetime = datetime.datetime.now() - datetime.timedelta(days=expireDay)
    expireTimestamp = time.mktime(expireDatetime.timetuple())
    activeFileList = sorted(glob.glob(reqActiveDir+'/*.log'), key=os.path.getmtime)
    for activeFile in activeFileList:
        activeFn = os.path.basename(activeFile)
        if os.path.isfile(activeFile):
            with open(activeFile, 'r') as f:
                try: reqPkgDict = yaml.load(f, Loader=yaml.SafeLoader)
                except: reqPkgDict = None
                
                if reqPkgDict != None:
                    packageRootDir = os.path.expanduser('~')+'/package'
                    for rf in reqPkgDict:
                        packagePath = os.path.normpath(packageRootDir+'/'+rf['dst'])
                        if packagePath.count('/package/') > 0 and os.path.isfile(packagePath):
                            try: os.unlink(packagePath)
                            except:
                                sympkgLogger.error('Delete error('+packagePath+').')
                                print 'Delete error('+packagePath+').'
                                continue

                            sympkgLogger.info('Delete '+packagePath+' ok.')
                            print 'Delete '+packagePath+' ok.'

        mtime = os.path.getmtime(activeFile)
        if mtime < expireTimestamp:
            print 'Active file expired.', activeFile
            activeFn = os.path.basename(activeFile)
            expireFile = reqExpiredDir+'/'+activeFn
            try:
                shutil.move(activeFile, expireFile)
                sympkgLogger.info('Active to Expire ok('+activeFile+').')
                print 'Active to Expire ok('+activeFile+').'
            except:
                sympkgLogger.error('Active to Expire error('+activeFile+').')
                print 'Active to Expire error('+activeFile+').'

def activeExpire():
    expireDatetime = datetime.datetime.now() - datetime.timedelta(days=8)
    expireTimestamp = time.mktime(expireDatetime.timetuple())
    activeFileList = sorted(glob.glob(reqActiveDir+'/*.log'), key=os.path.getmtime)
    for activeFile in activeFileList:
        try:
            os.utime(activeFile, (expireTimestamp, expireTimestamp))
            sympkgLogger.info('Active file expired('+activeFile+').')
            print 'Active file expired('+activeFile+').'
        except:
            sympkgLogger.error('Active Expire error('+activeFile+').')
            print 'Active Expire error('+activeFile+').'

def main():
    if not os.path.isdir(reqDir):
        sympkgLogger.error('Request directory is not exists('+reqDir+').')
        print 'Request directory is not exists.('+reqDir+')'
        return

    waitToActive()
    # activeExpire()
    # activeToExpired()

    print 'pollsympkg end.'

if __name__ == '__main__':
    main()
