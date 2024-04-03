import os, sys

b='hou.node("%s").render(frame_range=(%s,%s), verbose=False, output_progress=False)'%(sys.argv[2],sys.argv[3],sys.argv[4])
#render(self, frame_range=(), res=(), output_file=None,
#output_format=None, to_flipbook=False, quality=2, ignore_inputs=False,
#method=RopByRop or FrameByFrame, ignore_bypass_flags=False, ignore_lock_flags=False.
#verbose=False, output_progress=False)

hou.hipFile.load(os.path.expandvars(sys.argv[1]))
hou.hscript('setenv HIP=%s' % (sys.argv[6]) ) # for $JOB
hou.hscript('setenv JOB=%s' % (sys.argv[7]) ) # for $HIP
hou.hscript('varchange')

if sys.argv[9] == '0':
    hou.node(sys.argv[2]).parm('sopoutput').set(sys.argv[5][:-1]+sys.argv[8])
else:
    hou.node(sys.argv[2]).parm('sopoutput').set(sys.argv[5]+'$F4'+sys.argv[8])

exec b
