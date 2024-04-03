name = 'inventory'
version = '1.0.0'

tools = [
    'applyInventory',
    'createThumbnail'
]

def commands():
    env.PATH.append('{root}/bin')
    env.PYTHONPATH.append('{root}/scripts')

    env.INVENTORY_PATH.set('{root}')
