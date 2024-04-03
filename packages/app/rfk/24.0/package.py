name = 'rfk'
version = '24.0'

requires = [
    'renderman-24.0',
    'dxusd_katana'
]

variants = [
    ['katana-3.6'],
    ['katana-4.0']
]

def commands():
    import os

    version = '24.0'
    env.RFKTREE.set('/opt/pixar/RenderManForKatana-{}'.format(version))
    if not os.path.exists(str(env.RFKTREE)):
        stop('Not installed RenderManForKatana-{}'.format(version))

    env.KATANA_RESOURCES.append('{0}/plugins/katana{1}'.format(env.RFKTREE, getenv("REZ_KATANA_VERSION")))
    env.DEFAULT_RENDERER.set('prman')

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    alias('rfk', 'katana')
