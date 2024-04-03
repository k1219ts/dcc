#coding:utf-8
from __future__ import print_function

import sys
import getpass

import tractor.api.author as author
from tractor.TractorEngine import TractorEngine
TRACTOR_IP = '10.0.0.73'
PORT = 80

class JobMain:
    def __init__(self):
        pass

    def sampleTask(self, parent):
        task = author.Task(title='Render')
        parent.addChild(task)

        # command = ['rez-env', 'maya-2018', '--', 'mayapy']
        # command+= ['/dexter/Cache_DATA/CGSUP/sanghun.kim/WORK/scripts-develop/tractor/batchCmd.py']

        command = ['/backstage/dcc/DCC', 'mayapy', '/dexter/Cache_DATA/CGSUP/daeseok.chae/8_Scripts/test_Message.py']

        # # command = ['/backstage/dcc/DCC', 'rez-env', 'maya-2018', 'dxusd_maya', '--', 'mayapy']
        # command = ['rez-env', 'maya-2018', 'dxusd_maya', '--', 'mayapy']
        # command+= ['/backstage/dcc/packages/ext/dxusd_maya/1.0.0/scripts/DXBatch/BatchTest.py']

        task.addCommand(
            author.Command(argv=command)
        )


    def doIt(self):
        title = 'Script test sample.'
        job = author.Job(
            title=title,
            tier='cache',
            priority=100,
            projects=['export'],    # list
            service='Cache',
            comment='job command test',
            metadata=''
        )

        JobTask = author.Task(title='job')
        JobTask.serialsubtasks = 1
        job.addChild(JobTask)

        # test command
        self.sampleTask(JobTask)

        engine = TractorEngine(hostname=TRACTOR_IP, port=PORT, user=getpass.getuser(), debug=True)
        state, msg = engine.spool(job)
        print('# STATE')
        print('\t', state)
        print('# MESSAGE')
        print('\t', msg)


if __name__ == '__main__':
    JobMain().doIt()
