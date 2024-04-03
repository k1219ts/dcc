name = 'rfk'
version = '23.4'

requires = [
    'renderman-23.4',
    'dxusd_katana'
]

variants = [
    ['katana-3.5'],
    ['katana-3.6']
]

def commands():
    import os

    env.RFKTREE.set('/opt/pixar/RenderManForKatana-{0}-katana{1}'.format(version, getenv('REZ_KATANA_VERSION')))
    if not os.path.exists(str(env.RFKTREE)):
        stop('Not installed RenderManForKatana-{0}-katana{1}'.format(version, getenv('REZ_KATANA_VERSION')))

    env.KATANA_RESOURCES.append('{}/plugins/Resources/PRMan23'.format(env.RFKTREE))
    env.DEFAULT_RENDERER.set('prman')

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    alias('rfk', 'katana')
