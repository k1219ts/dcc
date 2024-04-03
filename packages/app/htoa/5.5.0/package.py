name = 'htoa'
version = '5.5.0'

variants = [
    ['houdini-18.5.351']
]

def commands():
    import os

    htoa_dir = '/opt/arnold/htoa-5.5.0.1_r45f2926_houdini-{}'.format(env.HVER)
    if not os.path.exists(str(htoa_dir)):
        stop('Not installed HoudiniToArnold-5.5.0')

    try:
        if env.SITE == "KOR":
            env.solidangle_LICENSE.set('5053@10.10.10.130')
        else:
            env.solidangle_LICENSE.set('5053@11.0.2.41')
    except:
        env.solidangle_LICENSE.set('5053@10.10.10.130')

    env.HOUDINI_PATH.append(htoa_dir)
