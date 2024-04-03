name = 'rfmari'
version = '22.6'

requires = [
    'mari-4.5+'
]

def commands():
    import os

    rfmDir = '/opt/pixar/RenderManForMari-22.6'
    if not os.path.exists(rfmDir):
        stop('Not installed RenderManForMari 22.6')

    env.MARI_SCRIPT_PATH.prepend(rfmDir)
