name = 'rfmari'
version = '23.2'

requires = [
    'mari-4.5+'
]

def commands():
    import os

    rfmDir = '/opt/pixar/RenderManForMari-23.2'
    if not os.path.exists(rfmDir):
        stop('Not installed RenderManForMari 23.2')

    env.MARI_SCRIPT_PATH.prepend(rfmDir)
    
