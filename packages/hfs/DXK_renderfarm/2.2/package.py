name = 'DXK_renderfarm'

def commands():
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')
    env.PYTHONPATH.append('{this.root}/scripts')
    # temporary use hou.config
    env.HOU_CONFIG_PATH.append('{this.root}/scripts')
