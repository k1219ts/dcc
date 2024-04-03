from config import *


class TractorEngine:
    '''
    Init Tractor-Engine
    param:
        - options = {'m_engine': str(ip), 'm_port': int(port), 'm_user': str(user)}
    '''
    def __init__(self, options, debug=True):
        self.tractorIP   = options['m_engine']
        self.tractorPort = options['m_port']
        self.user = options['m_user']

        self.client = EngineClient.EngineClient()
        self.client.setParam(
            hostname = self.tractorIP,
            port = self.tractorPort,
            user = self.user
        )

    def spool(self, job):
        try:
            result = self.client.spool(job.asTcl())
        except EngineClient.EngineClientError, err:
            print err
            return False

        resultDict = json.loads(result)

        # Debug
        print >> sys.stdout, time.ctime() + ' == > ' + '[%s]' % self.client.hostname + ' ' + resultDict['msg']
        return resultDict

    def __del__(self):
        self.client.close()
