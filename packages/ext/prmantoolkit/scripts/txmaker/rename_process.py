#!/user/bin/env python

import os, sys
import re, string
import fnmatch

# Mari to Mudbox
def mari2mud( filename ):
	baseName = os.path.basename( filename )
	udim = re.findall('\d+', baseName )
	if udim:
		if int( udim[-1] ) > 100:
			coord_base = int(udim[-1]) - 1000
			u_coord = coord_base - (coord_base/10) * 10
			v_coord = coord_base/10.0 + 1
			if v_coord == int(v_coord):
				u_coord = 10
				v_coord = v_coord - 1
			newName = filename.replace( udim[-1], 'u%s_v%d' % (u_coord, v_coord) )
			return newName

# Mudbox to Mari
def baseName_mud2mari( basename ):
	if fnmatch.fnmatch( basename, '*_u*_v*' ):
		source = re.compile('_u(\d+)_v(\d+)').findall(basename)
		if not source:
			return
		(u, v) = source[0]
		#(u, v) = re.compile('_u(\d+)_v(\d+)').findall(basename)[0]
		udim = (int(v) - 1) * 10
		udim += int(u)
		udim += 1000
		newName = basename.replace( '_u%s_v%s' % (u, v), '.%s' % str(udim) )
		return newName

