name = 'baselib'

requires = [
    'pylibs-2.7'
]

def commands():
    env.BACKSTAGE_PATH.set('/backstage')
    env.PYTHONPATH.prepend('/backstage/libs/python/2.7.16/lib/python2.7/site-packages')
    env.PYTHONPATH.append('/backstage/libs/tactic')
    env.PATH.prepend('/backstage/bin')
