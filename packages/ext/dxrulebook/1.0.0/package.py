name = 'dxrulebook'
version = '1.0.0'

tools = [
    'RulebookViewer.py'
]

variants = [
    ['python-3.10'],
    ['python-3.9'],
    ['python-3.7'],
    ['python-2']
]

def commands():
    env.PYTHONPATH.append('{root}/scripts')
    env.PATH.append('{root}/bin')
