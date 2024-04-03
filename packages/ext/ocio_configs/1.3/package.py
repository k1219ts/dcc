name = 'ocio_configs'
version = '1.3'

def commands():
    prefix = ''
    if 'nuke-14.0.6' in env.REZ_USED_RESOLVE.value():
        prefix = 'nuke_'
    elif 'katana' in env.REZ_USED_RESOLVE.value():
        prefix = 'katana_'
    elif 'nuke-13.2.5' in env.REZ_USED_RESOLVE.value():
        prefix = 'nuke13_'
    elif 'it' in env.REZ_USED_RESOLVE.value():
        prefix = 'it_'

    env.OCIO.set('{root}/%sconfig.ocio'% prefix)
