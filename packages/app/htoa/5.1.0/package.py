name = 'htoa'
version = '5.1.0'

variants = [
    ['houdini-17.5.460']
]

def commands():
    import os

    htoa_dir = '/opt/arnold/htoa-{VER}_r9289183_houdini-{HVER}'.format(VER=version, HVER=env.HVER)
    if not os.path.exists(str(htoa_dir)):
        stop('Not installed HoudiniToArnold-5.1.0')
    
    try:
        if env.SITE == "KOR":
            env.solidangle_LICENSE.set('5053@10.10.10.130')
        else:
            env.solidangle_LICENSE.set('5053@11.0.2.41')
    except:
        env.solidangle_LICENSE.set('5053@10.10.10.130')


    env.HOUDINI_PATH.append(htoa_dir)
