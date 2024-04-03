name = 'mtoa'
version = '5.2.1'

variants = [
    ['maya-2022']
]

def commands():
    import os

    mtoa_dir = '/opt/arnold/MtoA-5.2.1-linux-{}'.format(env.MAYA_VER)
    print mtoa_dir
    if not os.path.exists(str(mtoa_dir)):
        stop('Not installed MayaToArnold-5.2.1')

    try:
        if env.SITE == "KOR":
            env.solidangle_LICENSE.set('2080@10.10.10.171')
        else:
            env.solidangle_LICENSE.set('5053@11.0.2.41')
    except:
        env.solidangle_LICENSE.set('2080@10.10.10.171')

    env.MAYA_MODULE_PATH.append('{root}')
