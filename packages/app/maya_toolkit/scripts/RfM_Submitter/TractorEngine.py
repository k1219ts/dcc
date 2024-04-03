from config import *

class TractorEngine:
    def __init__( self, hostname=None, user=None, debug=True ):
        self.tractorIP = hostname
        self.tractorPort = GetEnginePort(self.tractorIP)

        print self.tractorIP
        print self.tractorPort

        self.client = EngineClient.EngineClient()
        self.client.setParam(
            hostname=self.tractorIP,
            port=self.tractorPort,
            user=user
        )
        self.getConfig()

    def getConfig(self):
        connector = TrHttpRPC(self.tractorIP, self.tractorPort)
        command = 'config?q=get&file=limits.config'
        limits = connector.Transaction(command, None, False)
        cfg = eval(limits[1])
        self.cfg_projects = list()
        if cfg['Limits'].has_key('ALL'):
            for i in cfg['Limits']['ALL']['Shares'].keys():
                if i.find('lt_') > -1:
                    self.cfg_projects.append(i[3:])

    def spool(self, job ):
        try:
            result = self.client.spool( job.asTcl() )
        except EngineClient.EngineClientError, err:
            print err
            return False

        resultDict = json.loads( result )

        # Debug
        print >> sys.stdout, time.ctime() + ' ==> ' + '[%s]' % self.client.hostname + ' ' + resultDict['msg']
        return resultDict

    def __del__( self ):
        self.client.close()
