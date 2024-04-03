import time
import json

import tractor.base.EngineClient as EngineClient

class TractorEngine:
    def __init__(self, hostname=None, port=80, user=None, debug=True):
        self.client = EngineClient.EngineClient()
        self.client.setParam(hostname=hostname, port=port, user=user)

    def spool(self, job):
        try:
            result = self.client.spool(job.asTcl())
            return True, json.loads(result)
        except EngineClient.EngineClientError, err:
            return False, err

    def __del__(self):
        self.client.close()
