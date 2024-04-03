name = 'maya_rigging'

# import platform
requires = [
    'jamong_maya'
]
# info = platform.uname()
# for pc in info :
#     if 'rig' in pc or 'ani' in pc :
#         requires.append('ricx_maya')

variants = [
    ['maya-2018'],
    ['maya-2022']
]

def commands():
    env.MAYA_MODULE_PATH.append('{root}')
