import os, sys
import copy
import argparse
from datetime import datetime, timedelta
from rocketchat.api import RocketChatAPI
import tractor.api.query as tq
from pymongo import MongoClient

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
client= MongoClient(DB_IP)
g_DB  = client['TractorMonitor_Test']
ERROR = g_DB['ERROR']

# Logging
from tmLogger import tmLogger
tmLog = tmLogger('TrError')


def SEND(bot, to, msg):
    api = RocketChatAPI(settings={'username': bot, 'password': 'dexterbot123', 'domain': 'http://10.10.10.232:3000'})
    return api.send_message(msg, to)


class Main:
    def __init__(self, engine, port):
        self.engine = engine
        self.port   = port
        self.ctime  = datetime(*datetime.now().timetuple()[:5])
        tq.setEngineClientParam(hostname=engine, port=port, user='editmasin')
        self.doIt()
        tq.closeEngineClient()

    def doIt(self):
        self.sendMessage()
        self.errorStatus()
        self.cleanUp()
        tmLog.info('process engine: %s port: %s' % (self.engine, self.port))


    def sendMessage(self):
        docs = ERROR.find({'engine': self.engine, 'pushed': False, '$and': [{'request_time': {'$lte': self.ctime}}]})
        if docs.count() == 0:
            return
        for d in docs:
            jobs = tq.jobs('error and jid=%s' % d['jid'])
            if jobs and not jobs[0]['pausetime']:
                msg = 'Error : tractor-engine %s jid %s\n' % (self.engine, d['jid'])
                msg+= jobs[0]['title']
                owner = jobs[0]['owner']
                result= SEND('BadBot', '@' + owner, msg)
                #result= SEND('BadBot', '@sanghun.kim', msg)
                if result and result['success']:
                    ERROR.update({'_id': d['_id']}, {'$set': {'pushed': True}})
                    tmLog.info('BadBot send : %s >> %s' % (msg, owner))
            else:
                ERROR.remove({'_id': d['_id']})
                tmLog.info('DB remove : %s' % d['jid'])
        tmLog.info('sendMessage : successed')


    def errorStatus(self):
        delaytime = 300 # 5m
        basis = {
            'engine': self.engine,
            'port': self.port,
            'time': self.ctime,
            'request_time': self.ctime + timedelta(seconds=delaytime),
            'pushed': False
        }
        errors = tq.jobs('error and spooltime > -2d', sortby=['jid'])
        if not errors:
            return

        docs = list()
        jids = list()
        for e in errors:
            if e['pausetime']:
                continue
            jid = e['jid']
            doc = ERROR.find_one({'jid': jid})
            if not doc:
                data = copy.copy(basis)
                data['jid'] = jid
                docs.append(data)
                jids.append(str(jid))
        if docs:
            ERROR.insert(docs)
            tmLog.info('DB insert : %s' % ','.join(jids))
        tmLog.info('errorStatus : successed')


    def cleanUp(self):
        docs = ERROR.find({'engine': self.engine, 'pushed': True, '$and': [{'request_time': {'$lte': self.ctime - timedelta(days=3)}}]})
        if docs.count() == 0:
            return
        jids = list()
        for d in docs:
            jids.append(str(d['jid']))
            ERROR.remove({'_id': d['_id']})
        tmLog.info('DB cleanUp : successed')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', type=str, default='10.0.0.30',
        help='tractor engine ip address')
    parser.add_argument('-p', '--port', type=int, default=80,
        help='tractor engine port number')

    args, unknown = parser.parse_known_args()
    Main(args.engine, args.port)
