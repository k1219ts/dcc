name = 'rfmari'
version = '24.4'

requires = [
    'mari-4.5+'
]

def commands():
    import os

    rfmDir = '/opt/pixar/RenderManForMari-24.4'
    if not os.path.exists(rfmDir):
        stop('Not installed RenderManForMari 24.4')

    env.MARI_SCRIPT_PATH.prepend(rfmDir)
