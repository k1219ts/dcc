name = 'tractor_monitor'

requires = [
    'python-2.7'
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
