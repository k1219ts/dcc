name = "DX_Environment"

requires = [
    'houdini-18.0+'
]

def commands():
    env.DX_ENVROOT.set('{this.root}')
    env.HOUDINI_OTLSCAN_PATH.append('{this.root}/otls')

def post_commands():
    # Debug
    import sys

    print 'DX_Environment Location : %s' % env.DX_ENVROOT
    print ''

