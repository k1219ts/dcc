name = 'HOU_Base'
version = '1.0.0'

variants = [
    ['houdini-18'],
    ['houdini-19']
]

def commands():
    env.HOUDINI_PATH.append('{root}')
    env.PYTHONPATH.append('{root}/scripts')
    env.HOUDINI_OTLSCAN_PATH.append('{root}/otls')

    # fx team common
    env.HOUDINI_PATH.append('/stdrepo/PFX/FXteamPath')
    env.PYTHONPATH.append('/stdrepo/PFX/FXteamPath/scripts')
    env.HOUDINI_OTLSCAN_PATH.append('/stdrepo/PFX/FXteamPath/otls')
    env.HOUDINI_VEX_PATH.append('/stdrepo/PFX/FXteamPath/scripts/vex:&')
    env.HOUDINI_OTLSCAN_PATH.append('/stdrepo/PFX/FXteamPath/append2Bundle/otls_test_lead_request')
    env.HOUDINI_MENU_PATH.append('/stdrepo/PFX/TD/scripts:$HH')