name = 'dxrunner'
version = '1.0.0'

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
    env.DXRUNNER_PATH.set('{root}')