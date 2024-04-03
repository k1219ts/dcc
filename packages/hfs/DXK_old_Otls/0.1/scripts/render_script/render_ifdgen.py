import os, sys
c='hou.node("%s").render(frame_range=(%s,%s), verbose=True, output_progress=True)'%(sys.argv[2],sys.argv[3],sys.argv[4])
#render(self, frame_range=(), res=(), output_file=None,
#output_format=None, to_flipbook=False, quality=2, ignore_inputs=False,
#method=RopByRop or FrameByFrame, ignore_bypass_flags=False, ignore_lock_flags=False.
#verbose=False, output_progress=False)

#sys.argv[1] = tmp Hip File
#sys.argv[2] = Mantra Node
#sys.argv[3] = Start Frame
#sys.argv[4] = End Frame
#sys.argv[5] = tmp Beauty Images Path
#sys.argv[6] = tmp Deep Images Path
#sys.argv[7] = ex) /show/ssss/shot/EB/EB_0010/fx/dev/scenes/houdini 
#sys.argv[8] = $JOB
#sys.argv[9] = Layer Name Folder Path

hou.hipFile.load(sys.argv[1])
hou.hscript('setenv HIP=%s' % (sys.argv[7]) ) # for $JOB
hou.hscript('setenv JOB=%s' % (sys.argv[8]) ) # for $HIP
hou.hscript('varchange')

hou.node(sys.argv[2]).parm('vm_picture').set(sys.argv[5])
if sys.argv[6] != 'None': # Render Deep images Check
   hou.node(sys.argv[2]).parm(sys.argv[9]).set(sys.argv[6])
#   hou.node(sys.argv[2]).parm('vm_dcmfilename').set(sys.argv[6])
hou.node(sys.argv[2]).setInput(0,None,0)

exec c
