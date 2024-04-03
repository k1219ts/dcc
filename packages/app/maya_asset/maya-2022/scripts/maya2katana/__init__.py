#!/usr/bin/python
"""
    maya2katana
    Copyright (C) 2016-2019 Andriy Babak, Animagrad

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Author: Andriy Babak
    e-mail: ababak@gmail.com
    ------------------------------
    Copy shader nodes to Katana
    ------------------------------
"""

__version__ = "3.3.4"

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import utils

reload(utils)

import clip

reload(clip)
copy = clip.copy
generate_xml = clip.generate_xml
del clip
