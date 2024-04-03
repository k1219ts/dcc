name = 'HOU_Feather'

def commands():
    env.HOUDINI_PATH.append('{this.root}')
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/hdas')
    env.HOUDINI_USER_PREF_DIR.append('{this.root}/presets')
