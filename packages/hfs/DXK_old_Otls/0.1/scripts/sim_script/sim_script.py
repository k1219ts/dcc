import os, sys, shutil
import subprocess
import threading
import Queue
import time
import datetime

import site
site.addsitedir( '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages' )
import tractor.api.author as author
import tractor.api.query as tq
import tractor.base.EngineClient as EngineClient

tq.setEngineClientParam(hostname = "10.0.0.30", port=80, user=sys.argv[15], debug=True)
jid = tq.tasks(("Job.owner=%s and Job.metadata='%s'") % (sys.argv[15], sys.argv[16]) , columns=['jid'])[0]['jid']
task = 'jid=%d and tid=%d' % (jid, 3) # 3 is first task node.
job = 'jid=%d' % jid

#########################################
# ... --> <hip file>_<user>_M_DD_HH_MM_S
#########################################
f_rendersc = sys.argv[0] # /netapp/fx_cache/_simulation/ifd/.../rendersc.py
f_tmppath  = sys.argv[1] # /tmp/HFS/.../cache
f_rootpath = sys.argv[2] # /netapp/fx_cache/test/scenes
f_tmproot  = sys.argv[3] # /tmp/HFS
f_hfs      = sys.argv[4] # hfs13.0.393
f_ifdgen   = sys.argv[5] # /netapp/fx_cache/_simulation/ifd/.../ifdgen.py
f_hipfile  = sys.argv[6] # /netapp/fx_cache/_simulation/hip/...hip
f_node     = sys.argv[7] # /obj/sphere/DX_HFS1/ropnet/OUT
f_start    = sys.argv[8] # 1
f_end      = sys.argv[9] # 20
f_duration = int(f_end) - int(f_start) + 1
f_output   = sys.argv[10] # /tmp/HFS/.../cache/test.
f_hip      = sys.argv[11] # $HIP
f_job      = sys.argv[12] # $JOB
f_tmpifd   = sys.argv[13] # /tmp/HFS/...
f_type     = int(sys.argv[14]) # single(0) or multi(1)

extFormat = sys.argv[17] # '.bgeo', '.bgeo.sc', '.vdb', 'abc' 
rangeType = sys.argv[18]



if rangeType == '0':
    f_lastframe= '%s%s' % (f_output[:-1], extFormat)

else:
    f_lastframe= '%s%04d%s' % (f_output, int(f_start) + int(f_duration) - 1, extFormat)


userName = sys.argv[15]
unique = sys.argv[16]


#########################################

sleeptime = 0.5 # 1sec for waiting next cache frame
queue = Queue.Queue()
ln1 = '-' * 150
ln2 = '*' * 150

#########################################
# SET ENV for INHOUSE LIB
#########################################
BROOT = '/netapp/backstage/pub'
HPATH = '%s/apps/houdini/build/%s' % (BROOT, f_hfs)
hfs_version = float(f_hfs.split('hfs')[-1][:4]) #16.0
f_hfs_root = HPATH
#if f_hfs.find('hfs15.5') == -1:
if hfs_version <= 15.0:
   f_hfs_root = '/opt/%s' % f_hfs

if os.path.exists(BROOT):
   os.environ['LD_LIBRARY_PATH']='%s/lib/extern/lib:%s/lib/zelos/lib' % (BROOT, BROOT)
   os.environ['HOUDINI_DSO_PATH']='%s/lib/zelos/houdini/14.0.291/dso:%s/houdini/dso' % (BROOT, f_hfs_root)
   os.environ['PYTHONPATH']='%s/lib/python_lib' % (BROOT)
else:
   print '<< WARNNING >> Not Exists : %s' % BROOT
#########################################
   
def interruptJob(job, e):
    now = gettime()
    try:
       tq.interrupt(job)
       print ln2
       print '<< ERROR >> Check Scene File. Rendering is Terminated...!!!!'
       print ' ** %s : Tractor Job Interrupted..!! : %s' % (now, job)
       print e
       print ln2
    except EngineClient.TransactionError, err:
       print ' Received exception for interrupting job - we should fix that'

def subcommand(cmd):    
    try:
       proc = subprocess.call(cmd, shell=True) #subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
       if proc > 0:
          print proc
          #interruptJob(job, proc)
    except OSError as e:
       interruptJob(job, e)

def gettime():
    now = datetime.datetime.now()
    result = '[%s]' % now.strftime('%H:%M:%S') #('%Y-%m-%d %H:%M:%S')
    return result

def copyfile(source, target):
    now = gettime()
    try:
       shutil.copy(source, target)
       #print '%s Copied : %s --> %s' % (now, os.path.basename(source), target)
       print '%s Copied : %s' % (now, target)
    except IOError, e:
       print ' ** %s Copy Error : %s - %s' % (now, e.filename, e.strerror)

def removefile(file):
    now = gettime()
    try: 
       os.remove(file)
       #print '%s Temp File Removed : %s' % (now, file)
    except IOError, e:
       print ' ** %s Remove Error : %s - %s' % (now, e.filename, e.strerror)

def cleanup():
    now = gettime()
    if os.path.exists(f_tmproot):
       if os.path.isdir(f_tmproot):
          try:
             shutil.rmtree(f_tmproot)
             #print '%s Temp Root Removed : %s' % (now, f_tmproot)             
          except OSError, e:
             print ' ** Error : %s - %s' % (now, e.filename, e.strerror)

def killproc(pname):
    cmd = 'ps -C %s -o pid=|xargs kill -9' % pname
    try:
       proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
       out, err = proc.communicate()
       proc.wait()
    except:
       pass    

class ThreadClass(threading.Thread):
    
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            f = self.queue.get()
            if f == 'render':               
               cmd = '%s/bin/hython "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s"' % (f_hfs_root, f_ifdgen, f_hipfile, f_node, f_start, f_end, f_output, f_hip, f_job, extFormat, rangeType)
               subcommand(cmd)

               # Copy Last Frame
               final_lastframe = os.path.join( f_rootpath, f_tmppath.split(os.sep)[-1], os.path.basename(f_lastframe) )
               if os.path.exists(f_lastframe) is True:
                  copyfile(f_lastframe, final_lastframe)
               self.queue.task_done()

            # copy and cleanup cache file.                  
            if os.path.exists(f) is True:
               final_path = os.path.join( f_rootpath, f_tmppath.split(os.sep)[-1] )
               final_file = os.path.join( final_path, os.path.basename(f) )
               copyfile(f, final_file)
               removefile(f)
               self.queue.task_done()
            else:
               time.sleep(sleeptime)

#########################################
# MAIN
#########################################
def main():
    print ln1
    print ' Job Overview'
    print ln1
    print '- HIP File to ifdgen : %s' % f_hipfile
    print '- Mantra Node : %s' % f_node
    print '- Start : %s' % f_start
    print '- End   : %s' % f_end
    if rangeType == '0':
        print '- Output File : %s%s' % (f_output[:-1], extFormat)
    else:
        print '- Output File : %s$F4%s' % (f_output, extFormat)
    print '- $HIP : %s' % f_hip
    print '- $JOB : %s' % f_job
    print '- Rendering Script : %s' % f_rendersc
    print '- Rendering Temp Path : %s' % f_tmppath
    print '- Rendering Root Path : %s' % f_rootpath
    print '- Temp Root Dir : %s' % f_tmproot
    print '- Houdini Version : %s' % f_hfs
    print '- Ifdgen Script : %s' % f_ifdgen
    print ln1

    ##############################
    # Cleanup tmp before rendering
    ##############################
    # /tmp/HFS
    ##############################
    if os.path.exists(f_tmproot):
       if os.path.isdir(f_tmproot):
          try:
             shutil.rmtree(f_tmproot)
          except OSError, e:
             print '--> Error: %s - %s' % (e.filename, e.strerror)

    ##############################
    # Make rendering tmp dir
    # /tmp/HFS/.../cache
    try: os.makedirs(f_tmppath, 0777)    
    except: pass

    ##############################
    # KILL ZOMBIE PROCESSES
    ##############################
    killproc('hython-bin')
    killproc('houdinifx-bin')
    killproc('maya.bin')
    killproc('Render')
    killproc('prman')

    ##############################
    # START HSERVER
    ##############################
    cmd_hserver = 'if [ ! `ps -C hserver -o pid=|xargs` ]; then %s/bin/hserver; fi' % f_hfs_root
    subcommand(cmd_hserver)

    ##############################
    # Check Multi Simulation without Threading
    ##############################
    if (f_type == 1): # multi simulation
       cmd = '%s/bin/hython "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s"' % (f_hfs_root, f_ifdgen, f_hipfile, f_node, f_start, f_end, f_output, f_hip, f_job, extFormat, rangeType)
       subcommand(cmd) 

       # Copy Frame
       if rangeType == '0':
           f  = '%s%s' % (f_output[:-1], extFormat)
       else:
           f  = '%s%04d%s' % (f_output, int(f_start), extFormat)


       final_path = os.path.join( f_rootpath, f_tmppath.split(os.sep)[-1] )
       final_file = os.path.join( final_path, os.path.basename(f) )
       copyfile(f, final_file)
       #removefile(f)
       
    elif (f_type == 0): # sigle simulation
       ##############################
       # Daemon Thread
       ##############################
       start = time.time()
   
       # Thread 1 - ifdgen, Thread 2 - copy and cleanup
       for i in range(2):
           th = ThreadClass(queue)
           th.setDaemon(True)
           th.start()

       num  = 1
       flag = 1
       c_frame = int(f_start) # 1001
       while True:
           if flag:
              queue.put('render')      
              flag = 0
       
           for j in range(f_duration):

              if rangeType == '0':
                 f  = '%s%s' % (f_output[:-1], extFormat)
                 f2 = '%s%s' % (f_output[:-1], extFormat)           
              else:
                 f  = '%s%04d%s' % (f_output, c_frame + j, extFormat)
                 f2 = '%s%04d%s' % (f_output, c_frame + j + 1, extFormat) 


              while True:
                 if num == int(f_duration): break
                 if os.path.exists(f2) is True:
                    queue.put(f)
                    num += 1
                    break
                 else:
                    time.sleep(sleeptime)
           # Auto closing
           break

       ##############################
       queue.join()
       print ln1
       print 'Elapsed Time : %s Sec' % (time.time() - start)
       print ln1

    ############################## 
    # remove /tmp/HFS
    ##############################
    cleanup()
    tq.closeEngineClient()

if __name__ == '__main__':
    main()
