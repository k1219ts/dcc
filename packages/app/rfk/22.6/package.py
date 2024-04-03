name = 'rfk'
version = '22.6'

requires = [
    'renderman-22.6'
]

variants = [
    ['katana-3.2']
]

def commands():
    import os

    env.RFKTREE.set('/opt/pixar/RenderManForKatana-22.6-katana3.2')
    if not os.path.exists(str(env.RFKTREE)):
        stop('Not installed RenderManForKatana-22.6-katana3.2')

    env.KATANA_RESOURCES.append('{}/plugins/Resources/PRMan22'.format(env.RFKTREE))
    env.DEFAULT_RENDERER.set('prman')

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    alias('rfk', 'katana')
