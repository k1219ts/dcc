#!/usr/bin/python

"""
The render script that's actually executed on the farm

TODO:
- need to integrate the split IFD generation/mantra tasks and cleanup
- at the moment it's using the China "unlimited" hbatch licenses for rendering

LAST RELEASE:
- 2017.07.09 $1 : submitter hipfile copy to local
- 2017.09.13 #2 : submitter hipfile direct render in server
                  direct call hython
"""

import hou
from config import *


class RenderJob(object):
    def __init__(self):

        print ("RENDER JOB STARTING")
        self.args = None
        self.wallClock = time.time()
        self.job = -1
        self.ifdFiles = []

        self.node = None

        self.RENDER_TMP = ""
        self.HOUDINI_VERSION = ""

        self.parseArguments()
        self.prnStartingInfo()

        self.run()


    def prnStartingInfo(self):
        # debug environment variable
        # pprint.pprint( os.environ )
        print ('-' * 150)
        print (' Rendering Job Overview')
        print ('-' * 150)
        print ('- HIP File to ifdgen : {hipfile}'.format(hipfile=self.args.hipFile))
        print ('- ROP Node : {mantraNode}'.format(mantraNode=self.args.ROPnode))
        print ('- startFrame : {startframe}'.format(startframe=self.args.startFrame))
        print ('- endFrame : {endframe}'.format(endframe=self.args.endFrame))
        print ('- User name : {username}'.format(username=self.args.userName))
        print ('- HFS : {hfs}'.format(hfs=os.getenv('HFS')))
        print ('- ENV : ', os.environ)
        # print '- OTLSCAN_PATH : {houdiniOtls}'.format(houdiniOtls=os.getenv('HOUDINI_OTLSCAN_PATH'))
        print ('-' * 150)

    def prnEndingInfo(self):
        sys.stderr.write('-' * 150 + '\n')
        sys.stderr.write(
            'Elapsed Time : {duration} Sec\n'.format(duration=time.time() - self.wallClock)
        )
        sys.stderr.write('-' * 150 + '\n')


    def parseArguments(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("-jobId", type=int, default=-1, help="Job ID")
        parser.add_argument("-hipFile", type=str, default="", help=".hip file to render")
        parser.add_argument("-ROPnode", type=str, default="", help="ROP Node to render")
        parser.add_argument("-startFrame", type=float, default=1.0, help="Start frame")
        parser.add_argument("-endFrame", type=float, default=1.0, help="End frame")
        parser.add_argument("-userName", type=str, default="", help="User name")

        self.args = parser.parse_args()

        self.job = 'jid={jobId}'.format(jobId=self.args.jobId)
        self.HOUDINI_VERSION = os.getenv('HOUDINI_VERSION')


    def run(self):
        # hipfile direct render in server
        try:
            hou.hipFile.load(str(self.args.hipFile), ignore_load_warnings=True)
            print ('Loaded: ', hou.hipFile.path())
            print ('Houdini version: ', hou.hscriptExpandString('$HOUDINI_VERSION'))
            self.render()
        except hou.LoadWarning:
            print ("Error: Not opend!")

        except Exception as e:#hou.LoadWarning:
            print ("run error : ", e.message)


    def render(self):
        self.node = hou.node(self.args.ROPnode)

        # disconnect inputs
        for x in range(len(self.node.inputConnections())):
            self.node.setInput(0, None)

        if self.node.type().description() == "Mantra":
            self.node.parm("soho_spoolrenderoutput").set(2)
            self.node.parm("vm_verbose").set(4)
            self.node.parm("vm_alfprogress").set(True)

        self.node.render(frame_range=(self.args.startFrame, self.args.endFrame),
                         verbose=True, output_progress=True)



if __name__ == '__main__':
    r = RenderJob()
    os._exit( 0 )
