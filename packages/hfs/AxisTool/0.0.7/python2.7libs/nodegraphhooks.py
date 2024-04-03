import hou
import nodegraph
from canvaseventtypes import *
from network_editor import list_node_dependencies

reload(list_node_dependencies)

## Double Clicking Nulls

def createEventHandler(uievent, pending_actions):
  if isinstance(uievent, MouseEvent) and uievent.eventtype == 'mousedown' and isinstance(uievent.selected.item, hou.Node):
    if uievent.selected.item.type().category().name() == 'Sop' and uievent.selected.item.type().name() == 'null':
      if len(pending_actions)==2:
        list_node_dependencies.startup(uievent.editor,uievent.selected.item)

  return None, False