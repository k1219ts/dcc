from utils import setvariable

## Set version number when hip saved by user

if kwargs["success"] and not kwargs["autosave"]:
    setvariable.version(kwargs['file'])