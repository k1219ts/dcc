name = 'nuke_scripts'
version = '3.0'

requires = [
    'dxrulebook'
]

def post_commands():
    env.NUKE_PATH.append('{root}/scripts')
