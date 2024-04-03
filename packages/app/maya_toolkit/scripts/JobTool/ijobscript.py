import os, sys, string, re
import maya.cmds as cmds


#-------------------------------------------------------------------------------
# Frame Iterate
def FrameIterateCount(frameRange, Engine, NodeCount):
    fr = []
    fRange = frameRange.split(',')
    for i in fRange:
        if len(i.split('-')) > 1:
            #   Sequence Frame
            frame = i.split('-')
            start_frame = int(frame[0])
            end_frame = int(frame[-1])

            hostbyframe = (end_frame - start_frame + 1) / NodeCount
            chk_point = hostbyframe * NodeCount + start_frame - 1

            if hostbyframe > 1:
                for i in range(NodeCount):
                    sf = start_frame + (i * hostbyframe)
                    ef = sf + hostbyframe - 1
                    if i == NodeCount - 1:
                        fr.append('%s-%s' % (sf, end_frame))
                    else:
                        fr.append('%s-%s' % (sf, ef))
            else:
                for i in range(start_frame, end_frame+1):
                    fr.append(str(i))
        else:
            fr.append(str(i))
    return fr


#-------------------------------------------------------------------------------
#
#   Maya (Software, Mental, Vray, Arnold)
#
#-------------------------------------------------------------------------------
def FrameIterate(opts):
    finfo = []
    fRange = opts['frameRange'].split(',')
    for i in fRange:
        if len(i.split('-')) > 1:
            # sequence frame
            frame = i.split('-')
            start_frame = int(frame[0])
            end_frame = int(frame[-1])
            finfo.append((start_frame, end_frame))
        else:
            finfo.append(int(i))
    return finfo

def FramePacketIterate(opts):
    finfo = []
    fRange = opts['frameRange'].split(',')
    for i in fRange:
        if len(i.split('-')) > 1:
            for j in range(int(i.split('-')[0]), int(i.split('-')[1]), int(opts['packet'])):
                if int(i.split('-')[1])-int(opts['packet']) > j:
                    finfo.append([j, j+int(opts['packet'])-1])
                else:
                    finfo.append([j, i.split('-')[1]])
        else:
            finfo.append([int(i), int(i)])
    return finfo


def FrameTask(renderer, opts, fmin, fmax, renderLayerNameV="", renderLayerProfileV=""):
    ths = {'sw':'n', 'mr':'rt', 'vray':'threads', 'rman':'n'}
    alf = ''

    if renderer == 'rman' :
        mayaRibCmd = '/usr/local/bin/Render -r %s -rd "%%D(%s)" -%s 0 -s %s -e %s' % \
                ('rib', opts['mayaOutDir'], ths[renderer], fmin, fmax)
        mayaPrmanCmd = '/opt/pixar/RenderManProServer-16.3/bin/prman'

        # Frame
        alf += '\t\tTask -title {Frame %s-%s} -cmds {\n' % (fmin, fmax)
        alf += '\t\tRemoteCmd {%s -proj "%%D(%s)" "%%D(%s)"}' % \
                (mayaRibCmd, opts['mayaProj'], opts['mayaScene'])
        alf += ' -atleast {1} -atmost {1} -samehost 1\n'

        renderLYRlist = cmds.ls(type="renderLayer")
        for renderLYRname in renderLYRlist:
            if cmds.getAttr(renderLYRname+'.renderable') :
                if renderLYRname == "defaultRenderLayer":
                    alf += '\t\tRemoteCmd {%s -t:0 -Progress -cwd "%%D(%s)" "%%D(%s)"}' % \
                            (mayaPrmanCmd, opts['mayaProj'], opts['mayaProj']+"/renderman/"+opts['mayaScene'].split("/")[-1].split(".")[0]+"/rib/"+"%04d"%fmin+"/"+"%04d"%fmin+".rib")
                else:
                    alf += '\t\tRemoteCmd {%s -t:0 -Progress -cwd "%%D(%s)" "%%D(%s)"}' % \
                            (mayaPrmanCmd, opts['mayaProj'], opts['mayaProj']+"/renderman/"+opts['mayaScene'].split("/")[-1].split(".")[0]+"/rib/"+"%04d"%fmin+"/"+"%04d"%fmin+"_"+renderLYRname+".rib")

                alf += ' -atleast {1} -atmost {1} -samehost 1\n'
        alf += '\t\t}\n'


    elif renderer == 'arnold' :
        cmdFlag = ""
        script_path = str(os.path.dirname(__file__))
        if opts["camera"] != "MultiCamera":
            cmdFlag += "-cam %s" % opts['camera']

        if cmdFlag != "":
            mayaCmd = '%s/bin/RenderCmd %s/Render -r %s %s -rl %s -rd "%%D(%s)" -ai:threads 0 -ai:lve 2 -s %s -e %s' % \
                      (script_path, opts['mayabin'], 'arnold', cmdFlag, renderLayerNameV, opts['mayaOutDir'], fmin, fmax)
        else:
            mayaCmd = '%s/bin/RenderCmd %s/Render -r %s -rl %s -rd "%%D(%s)" -ai:threads 0 -ai:lve 2 -s %s -e %s' % \
                      (script_path , opts['mayabin'], 'arnold', renderLayerNameV, opts['mayaOutDir'], fmin, fmax)

        # Frame
        alf += '\t\tTask -title {Layer %s %s-%s} -cmds {\n' % (renderLayerNameV, fmin, fmax)
        alf += '\t\tRemoteCmd {%s -proj "%%D(%s)" "%%D(%s)"}' % \
               (mayaCmd, opts['mayaProj'], opts['mayaScene'])
        #alf += ' -refersto %s\n' % renderLayerNameV
        alf += ' -service {%s}\n' % renderLayerProfileV
        alf += '\t\t}\n'

    else :
        if opts["camera"] != "MultiCamera":
            mayaCmd = '/usr/autodesk/maya%s/bin/Render -r %s -rl %s -cam %s -rd "%%D(%s)" -%s 0 -s %s -e %s' % \
                    (opts["mayaversion"], renderer, renderLayerNameV, opts['camera'], opts['mayaOutDir'], ths[renderer], fmin, fmax)
        else:
            mayaCmd = '/usr/autodesk/maya%s/bin/Render -r %s -rl %s -rd "%%D(%s)" -%s 0 -s %s -e %s' % \
                      (opts["mayaversion"], renderer, renderLayerNameV, opts['mayaOutDir'], ths[renderer], fmin, fmax)
        # Frame
        alf += '\t\tTask -title {Frame %s-%s} -cmds {\n' % (fmin, fmax)
        alf += '\t\tRemoteCmd {%s -proj "%%D(%s)" "%%D(%s)"}' % \
                (mayaCmd, opts['mayaProj'], opts['mayaScene'])
        alf += ' -atleast {1} -atmost {1} -samehost 1\n'
        alf += '\t\t}\n'
    return alf



def MayaMain(renderer, opts, jobscriptfile):
    print "Main :", renderer, opts, jobscriptfile
    alf = ''
    # Environment Key
    alf += '##AlfredToDo 3.0\n'
    alf += '\n'
    alf += 'Job -title {[%s] %s (%s)}' % (opts['showname'], opts['baseName'], renderer)
    alf += ' -comment { %s }' % opts["imgDirName"]
    show_name=""
    if "SHOW" in os.environ:
        show_name = os.environ["SHOW"]
    else:
        show_name = opts["showname"]
    alf += ' -projects { %s }' % show_name
    alf += ' -dirmaps {\n'
    alf += '\t{{mayabatch} {maya} NFS}\n'
    alf += '}'
    alf += ' -pbias 0 -serialsubtasks 1 -init {\n'
    alf += '} -subtasks {\n'
    alf += '\tTask -title {Job} -serialsubtasks 0 -subtasks {\n'

    # Frame Packet Iterate

    if opts['packet'] > 1:
        for camNameV, camFrameV in opts["cameraInfo"].items():
            opts['frameRange'] = camFrameV
            opts['camera'] = camNameV

            alf += '\t\tTask -title {Camera : %s : %s } -subtasks {\n' % (camNameV, opts['frameRange'])

            for renderLayerNameV, renderLayerProfileV in opts["renderLayer"].items():
                alf += '\t\tTask -title {Layer : %s } -subtasks {\n' % renderLayerNameV
                for f, p in FramePacketIterate(opts):   #Packet
                    #print f, p, 'f', 'p',FramePacketIterate(opts)
                    alf += FrameTask(renderer, opts, f, p, renderLayerNameV, renderLayerProfileV)
                alf += '\t\t}\n'
            alf += '\t\t}\n'
    # Frame Iterate
    else:
        for camNameV, camFrameV in opts["cameraInfo"].items():
            opts['frameRange'] = camFrameV
            opts['camera'] = camNameV

            alf += '\t\tTask -title {Camera : %s : %s } -subtasks {\n' % (camNameV, opts['frameRange'])

            for renderLayerNameV, renderLayerProfileV in opts["renderLayer"].items():
                alf += '\t\tTask -title {Layer : %s } -subtasks {\n' % renderLayerNameV
                for f in FrameIterate(opts):       #listframe : f
                    if type(f).__name__ == 'tuple':
                        for i in range(f[0], f[1]+1):   # no Packet single

                            # check By Frame
                            # ---------------------------------------
                            if int( opts['byFrame'] )> 1:
                                frameMin = f[0]
                                frameMax = f[1]
                                framRange = range(frameMin, frameMax+1)

                                byFrame = int(opts['byFrame'])

                                if  i not in framRange[ : : byFrame ] :
                                    continue
                            # ---------------------------------------
                            alf += FrameTask(renderer, opts, i, i, renderLayerNameV, renderLayerProfileV)
                    else:
                        alf += FrameTask(renderer, opts, f, f, renderLayerNameV, renderLayerProfileV)  # no Packet multi
                alf += '\t\t}\n'
            alf += '\t\t}\n'
    alf += '\t} -cmds {\n'
    alf += '\t\tRemoteCmd {echo "%s" } -service {%s}\n' %(opts['mayaOutDir'], opts['profile'])

    alf += '\t} -cleanup {\n'
    alf += '\t\tRemoteCmd {/bin/rm -f %%D(%s)} -service {%s}\n' % (opts['mayaScene'], opts['profile'])
    alf += '\t}\n'
    alf += '}\n'


    return alf


class arnold_AlfredScript:
    def __init__(self, opts):
        self.opts = opts
        print "> arnold_AlfredScript"
        layerName = ""
        if opts['makeRenderLayerJob']:
            layerName = "." + opts['renderLayer'].keys()[0]

        self.alf = ''
        self.script_file = os.path.join(opts['mayaProj'], 'tmp', 'alfscript', '%s_arnold%s.%s.alf' % (opts['imgName'], layerName, opts['nowtime'] ))
        self.alf_main()

    def alf_main(self):
        self.alf += MayaMain('arnold', self.opts, self.script_file)



