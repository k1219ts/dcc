name = 'axiom'

def commands():
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')
    env.HOUDINI_DSO_PATH.append('{this.root}/dso')
    env.PYTHONPATH.append('{this.root}/scripts')
    env.PATH.append('{this.root}/bin')
