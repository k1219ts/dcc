# Code for parameter template
hou_parm_template = hou.IntParmTemplate("vm_dcmcompression", "DCM Compression", 1, default_value=([4]), min=0, max=10, min_is_strict=False, max_is_strict=False, naming_scheme=hou.parmNamingScheme.Base1)
hou_parm_template.setConditional( hou.parmCondType.DisableWhen, "{ vm_deepresolver != camera }")
hou_parm_template.setHelp("Compression value between 0 and 10. Used to limit the number of samples which are stored in a lossy compression mode. The compression parameter applies to opacity values, and determines the maximum possible error in opacity for each sample. For compression greater than 0, the following relationship holds: OfError = 1/(2^(10-compression))")
hou_parm_template.setTags({"spare_category": "Deep Output"})
