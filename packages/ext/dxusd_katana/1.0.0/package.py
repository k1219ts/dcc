name = 'dxusd_katana'
version = '1.0.0'

variants = [
    ['katana-3.5'],
    ['katana-3.6'],
    ['katana-4.0'],
    ['katana-5.0'],
    ['katana-6.0']
]

requires = [
    'dxrulebook',
    'openexr'
]

def commands():
    env.KATANA_RESOURCES.append('{root}/Resources')
    env.LUA_PATH.append('{base}/lua/?.lua')
    if resolve.katana.version.major == 6:
        env.PYTHONPATH.prepend('{root}/scripts')
        env.PYTHONPATH.append('/opt/pixar/RenderManProServer-25.2/lib/python3.9/site-packages')

    else:
        env.PYTHONPATH.prepend('{base}/scripts')
        

    # env.PXR_PLUGINPATH_NAME.append('{base}/kind')
    env.FNPXR_PLUGINPATH.append('{base}/kind')
