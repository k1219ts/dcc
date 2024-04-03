name = 'nuke_extend'

#variants = [
#    ['nuke-10'],
#    ['nuke-11']
#]


def commands():
    import os

    try:
        if env.SITE == "KOR":
            pass
        else:
            env.NUKE_PATH.append('{root}/nuke-%s/scripts' % getenv("REZ_NUKE_MAJOR_VERSION"))
    except:
        pass

