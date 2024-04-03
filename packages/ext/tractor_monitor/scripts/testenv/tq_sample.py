import tractor.api.query as tq

TRACTOR_ENGINE = '10.0.0.30'

def PrintJobs(jobs):
    print 'total num >>', len(jobs)
    for j in jobs:
        print '{0} - {1}'.format(j['title'], j['jid'])

tq.setEngineClientParam(hostname=TRACTOR_ENGINE, port=80, user='editmasin')

# # pasued
# jobs = tq.jobs('pausetime > -1s', sortby=['-spooltime'])
# PrintJobs(jobs)

def errors():
    jobs = tq.jobs('error and spooltime > -2d', sortby=['jid'])
    for j in jobs:
        if j['pausetime']:
            continue
        print '{0} - {1}'.format(j['title'], j['jid'])
#errors()

def tasks():
    actives = tq.tasks('state=active', sortby=['jid'])
    print len(actives)
tasks()

tq.closeEngineClient()
