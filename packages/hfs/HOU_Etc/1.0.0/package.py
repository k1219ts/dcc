name = 'HOU_Etc'
version = '1.0.0'

variants = [
    ['houdini-18']
]

def commands():
    env.HOUDINI_PATH.append('{root}')
    env.PYTHONPATH.append('{root}/scripts')
    env.HOUDINI_OTLSCAN_PATH.append('{root}/otls')
