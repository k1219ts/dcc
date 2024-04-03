import os, sys, shutil
import subprocess
import time
import datetime

import site
site.addsitedir( '/netapp/backstage/pub/apps/tractor/linux/Tractor-2.2/lib/python2.7/site-packages' )
import tractor.api.author as author
import tractor.api.query as tq
import tractor.base.EngineClient as EngineClient

tq.setEngineClientParam(hostname = "10.0.0.30", port=80, user=sys.argv[16], debug=True)
jid = tq.tasks(("Job.owner=%s and Job.metadata='%s'") % (sys.argv[16], sys.argv[17]), columns=['jid'])[0]['jid']
task = 'jid=%d and tid=%d' % (jid, 3) # 3 is first task node.
job = 'jid=%d' % jid

#########################################
# ... --> <hip file>_<user>_M_DD_HH_MM_S
#########################################
f_rendersc = sys.argv[0] # /netapp/fx_cache/_simulation/ifd/.../rendersc.py
f_tmppath  = sys.argv[1] # /tmp/HFS/.../images
f_rootpath = sys.argv[2] # /netapp/fx_cache/test/images/layer1
f_tmproot  = sys.argv[3] # /tmp/HFS
f_hfs      = sys.argv[4] # hfs13.0.393
f_ifdgen   = sys.argv[5] # /netapp/fx_cache/_simulation/ifd/.../ifdgen.py
f_hipfile  = sys.argv[6] # /netapp/fx_cache/_simulation/hip/...hip
f_node     = sys.argv[7] # /obj/sphere/DX_HFS1/ropnet/OUT
f_start    = sys.argv[8] # 1
f_end      = sys.argv[9] # 20
f_duration = int(f_end) - int(f_start) + 1
f_output   = sys.argv[10] # /tmp/HFS/.../images/test.$F4.exr
f_deepout  = sys.argv[11] # /tmp/HFS/.../images/test_dsm.$F4.exr
f_hip      = sys.argv[12] # $HIP
f_job      = sys.argv[13] # $JOB
f_rootpathdeep = sys.argv[14] # /netapp/fx_cache/test/images/layer1/deep
f_deepsolv = sys.argv[18] # deepshadow is vm_dsmfilename, deepcamera is vm_dcmfilename
f_tmpifd   = sys.argv[15] # /tmp/HFS/.../manta1/test.$F4.ifd

userName = sys.argv[16]
unique = sys.argv[17]

#########################################

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
#   os.environ['PYTHONPATH']= '%s/lib/python_lib' % (BROOT)
   pythonPath = '%s/lib/python_lib' % (BROOT)
   pythonPath += ':/netapp/backstage/pub/apps/houdini/15.5/scripts'	
   os.environ['PYTHONPATH']= pythonPath

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

#########################################
# MAIN
#########################################
def main():
    print ln1
    print ' Rendering Job Overview'
    print ln1
    print '- HIP File to ifdgen : %s' % f_hipfile
    print '- Mantra Node : %s' % f_node
    print '- Start : %s' % f_start
    print '- End   : %s' % f_end
    print '- $HIP : %s' % f_hip
    print '- $JOB : %s' % f_job
    print '- Temp Root Dir : %s' % f_tmproot
    print '- Rendering Script : %s' % f_rendersc
    print '- Rendering Root Path : %s' % f_rootpath
    print '- Rendering Temp Path : %s' % f_tmppath
    print '- Output File : %s' % f_output
    if f_deepsolv != 'None':
       print '- Output Deep File : %s' % f_deepout
       print '- Deep Parameter : %s' % f_deepsolv
       print '- Rendering Root Deep Path : %s' % f_rootpathdeep
    print '- Houdini Version : %s' % f_hfs
    print '- IFD File to Render : %s' % f_tmpifd
    print ln1

    start = time.time()

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
    # Cleanup tmp before rendering
    ##############################
    # /tmp/HFR
    ##############################
    if os.path.exists(f_tmproot):
       if os.path.isdir(f_tmproot):
          try:
             shutil.rmtree(f_tmproot)
          except OSError, e:
             print '--> Error: %s - %s' % (e.filename, e.strerror)
             
    ##############################             
    # /tmp/HFS
    ##############################
    f_tmproot2 = os.sep.join(f_tmproot.split(os.sep)[:2])+'/HFS'  #/tmp/HFS
    if os.path.exists(f_tmproot2):
       if os.path.isdir(f_tmproot2):
          try:
             shutil.rmtree(f_tmproot2)
          except OSError, e:
             print '--> Error: %s - %s' % (e.filename, e.strerror)             

    ##############################
    # Make rendering tmp dir
    # /tmp/HFR/.../images/layer1

    try: os.makedirs(f_tmppath, 0777)
    except: pass

    ##############################
    # Make rendering tmp dir
    # /tmp/HFR/.../images/layer1/deep
    if f_deepsolv != 'None':
       try: os.makedirs(f_rootpathdeep, 0777)    
       except: pass

    ##############################
    # Make rendering ifd tmp dir
    # /tmp/HFR/.../ifd
    try: os.makedirs(os.path.dirname(f_tmpifd), 0777)    
    except: pass

    ##############################
    # ifdgen
    ############################## 
    cmd = '%s/bin/hython "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s"' % (f_hfs_root, f_ifdgen, f_hipfile, f_node, f_start, f_end, f_output, f_deepout, f_hip, f_job, f_deepsolv)
    subcommand(cmd)

    ##############################
    # render
    ##############################
    cmd = '%s/bin/mantra -V1 -a -f "%s"' % (f_hfs_root, f_tmpifd)
    subcommand(cmd)

    ##############################
    # copy 
    ##############################
    if os.path.exists(f_output) is True:
       copyfile( f_output, os.path.join(f_rootpath, os.path.dirname(f_output).split(os.sep)[-1], os.path.basename(f_output)) )
       #removefile( f_output )  
    if f_deepsolv != 'None':
       if os.path.exists(f_deepout) is True:
          copyfile( f_deepout, os.path.join(f_rootpath, f_rootpathdeep, os.path.basename(f_deepout)) )
          #removefile ( f_deepout )

    print ln1
    print 'Elapsed Time : %s Sec' % (time.time() - start)
    print ln1

    ############################## 
    # remove /tmp/HFR
    ##############################
    cleanup()
    tq.closeEngineClient()

if __name__ == '__main__':
    main()
