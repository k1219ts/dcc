name = 'DXUSD'

def commands():
    env.HOUDINI_PATH.append('{this.root}')
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')
    env.PYTHONPATH.append('{this.root}/scripts')
    # env.PYTHONPATH.append('{this.root}/soho')

    # env.HOUDINI_SOHO_PATH.append('{this.root}/soho')
