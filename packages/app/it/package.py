name = 'it'

requires = [
    'renderman-23.5',
    'ocio_configs-1.2'
]

def commands():
    env.OCIO.set('{}/it_config.ocio'.format(getenv('REZ_OCIO_CONFIGS_ROOT')))
