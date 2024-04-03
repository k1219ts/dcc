name = 'rfmari'
version = '23.3'

requires = [
    'mari-4.5+'
]

def commands():
    import os

    rfmDir = '/opt/pixar/RenderManForMari-23.3'
    if not os.path.exists(rfmDir):
        stop('Not installed RenderManForMari 23.3')

    env.MARI_SCRIPT_PATH.prepend(rfmDir)
