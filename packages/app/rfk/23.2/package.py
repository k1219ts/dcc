name = 'rfk'
version = '23.2'

requires = [
    'renderman-23.2'
]

variants = [
    ['katana-3.5']
]

def commands():
    import os

    env.RFKTREE.set('/opt/pixar/RenderManForKatana-23.2-katana3.5')
    if not os.path.exists(str(env.RFKTREE)):
        stop('Not installed RenderManForKatana-23.2-katana3.5')

    env.KATANA_RESOURCES.append('{}/plugins/Resources/PRMan23'.format(env.RFKTREE))
    env.DEFAULT_RENDERER.set('prman')

    env.KATANA_RESOURCES.append('{root}/Resources')
    env.PYTHONPATH.append('{root}/scripts')
    env.LUA_PATH.append('{root}/lua/?.lua')

    alias('rfk', 'katana')
