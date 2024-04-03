name = 'krita'
version = '5.0.6'

requires = [
    'qt-5.12.6',
]

def commands():
    import os

    env.PATH.append('/opt/krita')
    env.LD_LIBRARY_PATH.append('/opt/gcc-10.2.1/usr/lib64')
    env.LD_LIBRARY_PATH.append('/opt/atomicorp/atomic/root/lib64')
    env.LD_LIBRARY_PATH.append('/opt/atomicorp/atomic/root/usr/lib64')

    if not os.path.exists('/opt/krita/krita-{}-x86_64.appimage'.format(version)):
        stop('Not installed krita-{}'.format(version))

    if not os.path.exists('/opt/gcc-10.2.1/usr/lib64'):
        stop('Not installed gcc-10.2.1')

    if not os.path.exists('/opt/atomicorp/atomic/root/usr/lib64') and not os.path.exists('/opt/atomicorp/atomic/root/lib64'):
        stop('Not installed atomic-zlib-1.2.11')
