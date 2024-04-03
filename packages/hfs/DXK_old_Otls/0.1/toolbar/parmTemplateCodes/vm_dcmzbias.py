# Code for parameter template
hou_parm_template = hou.FloatParmTemplate("vm_dcmzbias", "DCM Z-Bias", 1, default_value=([0.001]), min=0, max=10, min_is_strict=False, max_is_strict=False, look=hou.parmLook.Regular, naming_scheme=hou.parmNamingScheme.Base1)
hou_parm_template.setConditional( hou.parmCondType.DisableWhen, "{ vm_deepresolver != camera }")
hou_parm_template.setHelp("Used in compression to merge together samples which are closer than the given threshold. Samples that are closer together than this bias value, are merged into a single sample and storad at the average z value of all the merged samples.")
hou_parm_template.setTags({"spare_category": "Deep Output"})
