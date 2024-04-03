name = 'baselib'

requires = [
    'pylibs-3.7'
]

def commands():
    env.BACKSTAGE_PATH.set('/backstage')
    env.PYTHONPATH.prepend('/backstage/libs/python/3.9.15/lib/python3.9/site-packages')
    env.PYTHONPATH.append('/backstage/libs/tactic')
    env.PATH.prepend('/backstage/bin')
