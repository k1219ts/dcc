name = 'DXK_renderfarm'

def commands():
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')
    env.PYTHONPATH.append('{this.root}/scripts')
    env.PATH.append('{this.root}/bin')
