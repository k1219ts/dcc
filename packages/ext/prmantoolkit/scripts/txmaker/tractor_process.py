import os
import getpass
import site
import re
import json

from PySide2 import QtWidgets

import tractor.api.author as author
import tractor.base.EngineClient as EngineClient


class JobMain:
    def __init__(self, files, opts):
        self.tr_engine = '10.0.0.25'
        self.tr_port   = 80
        self.srvkey    = 'PixarRender'
        self.user      = getpass.getuser()

        self.files = files
        self.opts  = opts

    def doIt(self):
        client = EngineClient.EngineClient()
        client.setParam(hostname=self.tr_engine, port=self.tr_port, user=self.user)

        job = self.jobscript()
        # print job.asTcl()
        try:
            result = client.spool(job.asTcl())
            msg = '[%s]' % client.hostname + ' ' + json.loads(result)['msg']
            QtWidgets.QMessageBox.information(QtWidgets.QWidget(), 'Tractor Spool', msg)
        except EngineClient.EngineClientError, err:
            QtWidgets.QMessageBox.critical(QtWidgets.QWidget(), 'Error', 'Tractor Spool Failed.\t')

        client.close()


    def jobscript(self):
        title = '(TX) %s' % os.path.dirname(self.files[0])

        job = author.Job(
            title=title, priority=200,
            projects=['convert'], tags=['3d'], maxactive=10
        )

        JobTask = author.Task(title='job')
        job.addChild(JobTask)

        for f in self.files:
            name = os.path.basename(f).replace('.', ' ')
            task = author.Task(title=name)
            JobTask.addChild(task)

            # command  = ['DCC', 'rez-env', 'txmaker', '--']
            command = ['/backstage/dcc/DCC', 'rez-env', 'prmantoolkit', '--']
            if self.opts.m_maptype == 'envlatl' or f.split('.')[-1] == 'hdr':
                command += self.opts.getEnvCommand()
                command += [f]
            else:
                command += self.opts.getCommand()
                command += [f, self.opts.getOutFilename(f)]

            task.addCommand(
                author.Command(service=self.srvkey, argv=command)
            )

        return job
