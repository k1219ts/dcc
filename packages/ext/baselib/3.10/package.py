name = 'baselib'

requires = [
    'pylibs-3.10'
]

def commands():   

    env.BACKSTAGE_PATH.set('/backstage')
    env.PYTHONPATH.prepend('/backstage/libs/python/3.10.13/lib/python3.10/site-packages')
    env.PYTHONPATH.append('/backstage/libs/tactic')
    env.PATH.prepend('/backstage/bin')
