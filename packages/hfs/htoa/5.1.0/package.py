name = 'htoa'

variants = [
    ['houdini-17.5.460']
]

def commands():
    htoa_dir = '/opt/arnold/htoa-{VER}_r9289183_houdini-{HVER}'.format(VER=version, HVER=env.HVER)
    env.solidangle_LICENSE.set('5053@10.10.10.130')

    env.HOUDINI_PATH.append(htoa_dir)
