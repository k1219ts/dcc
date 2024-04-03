"""
NAME: Variant Viewer
ICON: /backstage/share/icons/pxr_usd.png
KEYBOARD_SHORTCUT: Ctrl+G
SCOPE:

USD Variant Menu to all

"""
import UI4
from VariantMenu.VariantMenuTab import VariantMenuMain

widget = VariantMenuMain(UI4.App.Tabs.FindTopTab("Node Graph").window())
