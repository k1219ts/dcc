name = 'DXK_old_Otls'

def commands():
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')
    env.HOUDINI_TOOLBAR_PATH.append('{this.root}/toolbar')
    env.PYTHONPATH.append('{this.root}/scripts')
