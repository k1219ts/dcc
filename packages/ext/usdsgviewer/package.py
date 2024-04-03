name = 'usdsgviewer'

requires = [
    'usd_core-21.05',
    'pyside2-5.12.6'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
