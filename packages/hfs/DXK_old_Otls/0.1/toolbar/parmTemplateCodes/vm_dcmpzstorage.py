# Code for parameter template
hou_parm_template = hou.StringParmTemplate("vm_dcmpzstorage", "DCM Z Storage", 1, default_value=(["real32"]), naming_scheme=hou.parmNamingScheme.Base1, string_type=hou.stringParmType.Regular, menu_items=(["real16","real32","real64"]), menu_labels=(["16 bit float","32 bit float","64 bit float"]), icon_names=([]), item_generator_script="", item_generator_script_language=hou.scriptLanguage.Python, menu_type=hou.menuType.Normal)
hou_parm_template.setConditional( hou.parmCondType.DisableWhen, "{ vm_deepresolver != camera }")
hou_parm_template.setHelp("Specifies the amount of bits to use to store opacity samples. The default is 32 bits. Smaller values may cause unnecessary discretization of samples of sample far away from the camera, but can save substantially on file size.")
hou_parm_template.setTags({"spare_category": "Deep Output"})
