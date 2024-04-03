name = 'rfmari'
version = '25.2'

requires = [
    'mari-6.0+'
]

def commands():
    import os

    rfmDir = '/opt/pixar/RenderManForMari-25.2'
    if not os.path.exists(rfmDir):
        stop('Not installed RenderManForMari 25.2')

    env.MARI_SCRIPT_PATH.prepend(rfmDir)
