name = 'prmantoolkit'

requires = [
    'baselib-2.5',
    'renderman-23.5',
    'pyside2-5.12.6'
]

tools = [
    'txmaker',
    'txenvlatl',
    'it',
    'LocalQueue'
]

def commands():
    env.PATH.prepend('{root}/bin')
