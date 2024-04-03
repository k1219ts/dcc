"""
Setting up the Tractor connection
"""

import config
from config import *


class TractorEngine:
    def __init__(self, renderer=None):
        self.user = os.getlogin()
        self.tractorIP   = config.getConfig['TractorEngine'][0]
        self.tractorPort = int(config.getConfig['TractorEngine'][1])
        if renderer == 'redshift':
            self.tractorIP   = '10.0.0.25'
            self.tractorPort = 80

        self.client = EngineClient.EngineClient()
        self.client.setParam(
            hostname=self.tractorIP,
            port=self.tractorPort,
            user=self.user)
        self.getConfig()

    def getConfig(self):
        connector = TrHttpRPC(self.tractorIP, self.tractorPort)
        command   = 'config?q=get&file=limits.config'
        limits = connector.Transaction(command, None, False)
        cfg = eval(limits[1])
        self.cfg_projects = list()
        for i in cfg['Limits']['ALL']['Shares'].keys():
            if i.find('fx_') > -1:
                self.cfg_projects.append(i[3:])

    def spool(self, job):
        try:
            result = self.client.spool(job.asTcl())
        except EngineClient.EngineClientError, err:
            print err
            return False

        resultDict = json.loads(result)

        # Debug
        print >> sys.stdout, time.ctime() + ' ==> ' + '[%s]' % self.client.hostname + ' ' + resultDict['msg']
        return resultDict

    def __del__(self):
        self.client.close()
