name = 'loopABC'

requires = [
    'houdini-18.5.351',
    'pyalembic-1.7.1',
    'hdf5'
]

def commands():
    env.PATH.prepend('{root}/bin')
    env.LOOPABC_SOURCEPATH.set('{root}/source')
