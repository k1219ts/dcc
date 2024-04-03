'''
Dexter Tractor Engine Setup
'''
import json
import time
import getpass

import tractor.base.EngineClient as EngineClient

class Engine:
    def __init__(self, hostname='10.0.0.25', port=80, user=getpass.getuser(), debug=True):
        self.client = EngineClient.EngineClient()
        self.client.setParam(hostname=hostname, port=port, user=user)
        self.debug = debug

    def spool(self, job):
        try:
            result = self.client.spool(job.asTcl())
        except EngineClient.EngineClientError, err:
            print err
            return False

        resultDict = json.loads(result)

        # Debug
        if self.debug:
            result_message = time.ctime() + '==> ' + '[%s]' % self.client.hostname + ' ' + resultDict['msg']
            print '[INFO TractorSpool Message] :', result_message
        return resultDict

    def __del__(self):
        self.client.close()
