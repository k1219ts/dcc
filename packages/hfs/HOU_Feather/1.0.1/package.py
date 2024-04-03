name = 'HOU_Feather'

variants = [
    ['houdini-18']
]

def commands():
    env.HOUDINI_PATH.append('{root}')
    env.HOUDINI_OTLSCAN_PATH.append('{root}/otls')
    env.PYTHONPATH.append('{root}/scripts')
