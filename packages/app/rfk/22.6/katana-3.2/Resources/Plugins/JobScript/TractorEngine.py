from Katana import UI4

import os
import time
import configobj
import site
import json

import dxConfig
site.addsitedir(dxConfig.getConf('TRACTOR_API'))

import tractor.base.EngineClient as EngineClient

class TractorEngine:
    def __init__(self, hostname=None, port=80, user=None, debug=True):
        self.client = EngineClient.EngineClient()
        self.client.setParam(hostname=hostname, port=port, user=user)

    def spool(self, job):
        try:
            result = self.client.spool(job.asTcl())
        except EngineClient.EngineClientError, err:
            print err
            UI4.Widgets.MessageBox.Critical('Tractor Spool', err)
            return False

        resultDict = json.loads(result)

        # Debug
        result_message = time.ctime() + ' ==> ' + '[%s]' % self.client.hostname + ' ' + resultDict['msg']
        print '[INFO TractorSpool Message] :', result_message
        UI4.Widgets.MessageBox.Information('Tractor Spool', result_message)
        return resultDict

    def __del__(self):
        self.client.close()
