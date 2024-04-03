import os, sys
import copy
import argparse
from datetime import datetime, timedelta
import tractor.api.query as tq
from pymongo import MongoClient

import dxConfig
DB_IP = dxConfig.getConf('DB_IP')
client= MongoClient(DB_IP)
g_DB  = client['TractorMonitor_Test']
LIVE  = g_DB['LIVE']

# Logging
from tmLogger import tmLogger
tmLog = tmLogger('TrLive')


def procNameByLimits(limits):
    if 'katanarender' in limits and not 'denoise' in limits:
        return 'katana'
    elif 'denoise' in limits:
        return 'denoise'
    elif 'houdini' in limits and not 'rm' in limits:
        return 'houdini'
    else:
        return 'other'

class commandInfo:
    def __init__(self, jid, cid):
        self.argv = tq.commands('jid=%s and cid=%s' % (jid, cid))[0]['argv']
        # result variable
        self.proc = None
        self.show = 'other'
        self.shot = 'other'

        self.doIt()

    def doIt(self):
        self.proc = self.getProcess()
        eval('self.%s_parser()' % self.proc)

    def get_show(self):
        return self.show

    def getProcess(self):
        proc = 'other'
        if 'katana' in self.argv:
            proc = 'katana'
        if 'denoise' in self.argv:
            proc = 'denoise'
        if '-hipFile' in self.argv:
            proc = 'houdini'
        return proc

    def getShowName(self, src):
        show = 'other'
        if 'show' in src:
            show = src[src.index('show')+1]
            show = show.replace('_pub', '')
        return show


    def katana_parser(self):
        katfile = self.argv[self.argv.index('katana')+2].split('=')[-1]
        src = katfile.split('/')
        self.show = self.getShowName(src)
        for s in src:
            if s.startswith('shot_'):
                shot = s.split('_seq')[0].replace('shot_', '')
                self.shot = shot
                break

    def denoise_parser(self):
        src = self.argv[-1].split('/')
        self.show = self.getShowName(src)
        if 'shot' in src:
            shot = src[src.index('shot') + 2]
            self.shot = shot

    def houdini_parser(self):
        hipfile = self.argv[self.argv.index('-hipFile') + 1]
        src = hipfile.split('/')
        self.show = self.getShowName(src)
        if 'shot' in src:
            shot = src[src.index('shot') + 2]
            self.shot = shot
        elif 'asset' in src:
            self.shot = 'asset'

    def other_parser(self):
        self.denoise_parser()


#-------------------------------------------------------------------------------
class Main:
    def __init__(self, engine, port):
        self.engine = engine
        self.port   = port

        self.ctime = datetime(*datetime.now().timetuple()[:5])
        self.basis = {
            'engine': engine, 'port': port, 'time': self.ctime
        }

        tq.setEngineClientParam(hostname=engine, port=port, user='editmasin')
        self.doIt()
        tq.closeEngineClient()

    def doIt(self):
        last = LIVE.find_one({'state': 'active'}, sort=[('time', -1)])

        self.activeTasks()


    def activeTasks(self):
        actives = tq.tasks('state=active', columns=['limits'], sortby=['jid'])
        if not actives:
            return

        proc_data = {
            'katana': [], 'houdini': [], 'denoise': []
        }
        job_data = {}   # {jid: [cid, ...]}

        for task in actives:
            jid = task['jid']
            pcn = procNameByLimits(task['Invocation.limits'])   # process name
            for cid in task['cids']:
                id = '{JID}-{TID}-{CID}'.format(JID=jid, TID=task['tid'], CID=cid)
                proc_data[pcn].append(id)
                if not job_data.has_key(jid):
                    job_data[jid] = list()
                job_data[jid].append(cid)
        jids = sorted(job_data.keys())

        jidShow = dict()    # {jid: 'srh', jid: 'ban'}
        for jid in jids:
            jidShow[jid] = commandInfo(jid, job_data[jid][0]).get_show()

        showNum = dict()    # {'srh': 29, 'ban': 230}
        for jid, show in jidShow.items():
            num = len(job_data[jid])
            if not showNum.has_key(show):
                showNum[show] = 0
            showNum[show] += num

        showProc = dict()   # {'srh.katana': 20, 'srh.houdini': 10}
        for proc, ids in proc_data.items():
            for id in ids:
                jid = id.split('-')[0]
                show= jidShow[int(jid)]
                key = '{SHOW}.{PROC}'.format(SHOW=show, PROC=proc)
                if not showProc.has_key(key):
                    showProc[key] = 0
                showProc[key] += 1

        # result
        data = copy.copy(self.basis)
        data['state']= 'active'
        data['proc'] = proc_data
        data['jids'] = jids
        data['show'] = showNum
        data['showproc'] = showProc

        simple_data = copy.copy(self.basis)
        simple_data['state']= 'sp_active'
        simple_data['show'] = showNum
        simple_data['showproc'] = showProc

        LIVE.insert([data, simple_data], check_keys=False)
        tmLog.info('DB push active tasks')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', type=str, default='10.0.0.30',
        help='tractor engine ip address')
    parser.add_argument('-p', '--port', type=int, default=80,
        help='tractor engine port number')

    args, unknown = parser.parse_known_args()
    Main(args.engine, args.port)
