name = 'HOU_Base'
version = '1.1.0'

variants = [
    ['houdini-18'],
    ['houdini-19'],
    ['houdini-20']
]

def commands():
    env.HOUDINI_PATH.append('{root}')
    env.PYTHONPATH.append('{root}/scripts')
    env.HOUDINI_OTLSCAN_PATH.append('{root}/otls')

    # fx team common
    env.HOUDINI_PATH.append('/stdrepo/PFX/FXteamPath/hou_bundle_19')
    env.PYTHONPATH.append('/stdrepo/PFX/FXteamPath/hou_bundle_19/scripts')
    env.HOUDINI_OTLSCAN_PATH.append('/stdrepo/PFX/FXteamPath/hou_bundle_19/otls')
    env.HOUDINI_VEX_PATH.append('/stdrepo/PFX/FXteamPath/hou_bundle_19/scripts/vex:&')
    env.HOUDINI_MENU_PATH.append('/stdrepo/PFX/TD/scripts:$HH')
