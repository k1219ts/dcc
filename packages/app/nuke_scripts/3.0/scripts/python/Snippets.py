'''
nuke for snippets _by jh
'''

import nuke

def make_Dot():
    for i in nuke.selectedNodes():
        out_node = i.input(0)
        dot_node = nuke.nodes.Dot()
        if out_node:
            dot_node.setInput(0, i)
            out_node.setInput(0, dot_node)
            if i.dependent():
                for dep_node in i.dependent():
                    dep_node.setInput(0, dot_node)
        else:
            dot_node.setInput(0, i)
            if i.dependent():
                for dep_node in i.dependent():
                    dep_node.setInput(0, dot_node)
