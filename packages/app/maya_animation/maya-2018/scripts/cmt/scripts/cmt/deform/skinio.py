"""Exports and imports skin weights.

Usage:
    Select a mesh and run

    # To export
    import skinio
    skinio.export_skin(file_path='/path/to/data.skin')

    # To import
    skinio.import_skin(file_path='/path/to/data.skin')
"""
import cPickle as pickle
import logging
from functools import partial

from PySide import QtGui
from maya.app.general.mayaMixin import MayaQWidgetBaseMixin

import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import maya.OpenMayaAnim as OpenMayaAnim

import cmt.shortcuts as shortcuts
logger = logging.getLogger(__name__)
EXTENSION = '.skin'


def import_skin(file_path=None, shapes=None, to_selected_shapes=False):
    """Creates a skinCluster on the specified shape if one does not already exist
    and then import the weight data.
    """
    selected_shapes = cmds.ls(sl=True) if to_selected_shapes else None

    if file_path is None:
        file_path = cmds.fileDialog2(dialogStyle=2, fileMode=1, fileFilter='Skin Files (*{0})'.format(EXTENSION))
    if not file_path:
        return
    if not isinstance(file_path, basestring):
        file_path = file_path[0]

    # Read in the file
    fh = open(file_path, 'rb')
    data = pickle.load(fh)
    fh.close()

    if shapes and not isinstance(shapes, basestring):
        shapes = [shapes, ]

    for skin_data in data:
        shape = skin_data['shape']
        if not cmds.objExists(shape):
            continue
        if selected_shapes and shape not in selected_shapes:
            continue
        elif not to_selected_shapes and shapes and shape not in shapes:
            continue

        # Make sure the vertex count is the same
        mesh_vertex_count = cmds.polyEvaluate(shape, vertex=True)
        imported_vertex_count = len(skin_data['blendWeights'])
        if mesh_vertex_count != imported_vertex_count:
            raise RuntimeError('Vertex counts do not match. %d != %d' %
                               (mesh_vertex_count, imported_vertex_count))

        # Check if the shape has a skinCluster
        skins = get_skin_clusters(shape)
        if skins:
            skin_cluster = SkinCluster(skins[0])
        else:
            # Create a new skinCluster
            joints = skin_data['weights'].keys()

            # Make sure all the joints exist
            unused_imports = []
            no_match = set([shortcuts.remove_namespace_from_name(x) for x in cmds.ls(type='joint')])
            for j in joints:
                if j in no_match:
                    no_match.remove(j)
                else:
                    unused_imports.append(j)
            # If there were unmapped influences ask the user to map them
            if unused_imports and no_match:
                mapping_dialog = WeightRemapDialog()
                mapping_dialog.set_influences(unused_imports, no_match)
                mapping_dialog.exec_()
                for src, dst in mapping_dialog.mapping.items():
                    # Swap the mapping
                    skin_data['weights'][dst] = skin_data['weights'][src]
                    del skin_data['weights'][src]

            # Create the skinCluster with post normalization so setting the weights does not
            # normalize all the weights
            joints = skin_data['weights'].keys()
            skin = cmds.skinCluster(joints, shape, tsb=True, nw=2, n=skin_data['name'])[0]
            skin_cluster = SkinCluster(skin)
        skin_cluster.set_data(skin_data)
        logging.info('Imported %s', file_path)


def get_skin_clusters(nodes):
    """Get the skinClusters attached to the specified node and all nodes in descendents.

    :param nodes: List of dag nodes.
    @return A list of the skinClusters in the hierarchy of the specified root node.
    """
    if isinstance(nodes, basestring):
        nodes = [nodes, ]
    all_skins = []
    for node in nodes:
        relatives = cmds.listRelatives(node, ad=True, path=True) or []
        relatives.insert(0, node)
        relatives = [shortcuts.get_shape(node) for node in relatives]
        for relative in relatives:
            history = cmds.listHistory(relative, pruneDagObjects=True, il=2) or []
            skins = [x for x in history if cmds.nodeType(x) == 'skinCluster']
            if skins:
                all_skins.append(skins[0])
    return list(set(all_skins))


def export_skin(file_path=None, shapes=None):
    """Exports the skinClusters of the given shapes to disk in a pickled list of skinCluster data.

    :param file_path: Path to export the data.
    :param shapes: Optional list of dag nodes to export skins from.  All descendent nodes will be searched for
    skinClusters also.
    """
    if shapes is None:
        shapes = cmds.ls(sl=True) or []

    # If no shapes were selected, export all skins
    skins = get_skin_clusters(shapes) if shapes else cmds.ls(type='skinCluster')
    if not skins:
        raise RuntimeError('No skins to export.')

    if file_path is None:
        file_path = cmds.fileDialog2(dialogStyle=2, fileMode=0, fileFilter='Skin Files (*{0})'.format(EXTENSION))
        if file_path:
            file_path = file_path[0]
    if not file_path:
        return
    if not file_path.endswith(EXTENSION):
        file_path += EXTENSION

    all_data = []
    for skin in skins:
        skin = SkinCluster(skin)
        data = skin.gather_data()
        all_data.append(data)
        logging.info('Exporting skinCluster %s (%d influences, %d vertices)',
                     skin.node, len(data['weights'].keys()), len(data['blendWeights']))
    fh = open(file_path, 'wb')
    pickle.dump(all_data, fh, pickle.HIGHEST_PROTOCOL)
    fh.close()


class SkinCluster(object):

    attributes = ['skinningMethod', 'normalizeWeights', 'dropoffRate', 'maintainMaxInfluences', 'maxInfluences',
                  'bindMethod', 'useComponents', 'normalizeWeights', 'weightDistribution', 'heatmapFalloff']

    def __init__(self, skin_cluster):
        """Constructor"""
        self.node = skin_cluster
        self.shape = cmds.listRelatives(cmds.deformer(skin_cluster, q=True, g=True)[0], parent=True, path=True)[0]

        # Get the skinCluster MObject
        self.mobject = shortcuts.get_mobject(self.node)
        self.fn = OpenMayaAnim.MFnSkinCluster(self.mobject)
        self.data = {
            'weights': {},
            'blendWeights': [],
            'name': self.node,
            'shape': self.shape
        }

    def gather_data(self):
        """Gather all the skinCluster data into a dictionary so it can be serialized.

        :return: The data dictionary containing all the skinCluster data.
        """
        dag_path, components = self.__get_geometry_components()
        self.gather_influence_weights(dag_path, components)
        self.gather_blend_weights(dag_path, components)

        for attr in SkinCluster.attributes:
            self.data[attr] = cmds.getAttr('%s.%s' % (self.node, attr))
        return self.data

    def __get_geometry_components(self):
        """Get the MDagPath and component MObject of the deformed geometry.

        :return: (MDagPath, MObject)
        """
        # Get dagPath and member components of skinned shape
        fnset = OpenMaya.MFnSet(self.fn.deformerSet())
        members = OpenMaya.MSelectionList()
        fnset.getMembers(members, False)
        dag_path = OpenMaya.MDagPath()
        components = OpenMaya.MObject()
        members.getDagPath(0, dag_path, components)
        return dag_path, components

    def gather_influence_weights(self, dag_path, components):
        """Gathers all the influence weights

        :param dag_path: MDagPath of the deformed geometry.
        :param components: Component MObject of the deformed components.
        """
        weights = self.__get_current_weights(dag_path, components)

        influence_paths = OpenMaya.MDagPathArray()
        influence_count = self.fn.influenceObjects(influence_paths)
        components_per_influence = weights.length() / influence_count
        for ii in range(influence_paths.length()):
            influence_name = influence_paths[ii].partialPathName()
            # We want to store the weights by influence without the namespace so it is easier
            # to import if the namespace is different
            influence_without_namespace = shortcuts.remove_namespace_from_name(influence_name)
            self.data['weights'][influence_without_namespace] = \
                [weights[jj*influence_count+ii] for jj in range(components_per_influence)]

    def gather_blend_weights(self, dag_path, components):
        """Gathers the blendWeights

        :param dag_path: MDagPath of the deformed geometry.
        :param components: Component MObject of the deformed components.
        """
        weights = OpenMaya.MDoubleArray()
        self.fn.getBlendWeights(dag_path, components, weights)
        self.data['blendWeights'] = [weights[i] for i in range(weights.length())]

    def __get_current_weights(self, dag_path, components):
        """Get the current skin weight array.

        :param dag_path: MDagPath of the deformed geometry.
        :param components: Component MObject of the deformed components.
        :return: An MDoubleArray of the weights.
        """
        weights = OpenMaya.MDoubleArray()
        util = OpenMaya.MScriptUtil()
        util.createFromInt(0)
        ptr = util.asUintPtr()
        self.fn.getWeights(dag_path, components, weights, ptr);
        return weights

    def set_data(self, data):
        """Sets the data and stores it in the Maya skinCluster node.

        :param data: Data dictionary.
        """

        self.data = data
        dag_path, components = self.__get_geometry_components()
        self.set_influence_weights(dag_path, components)
        self.set_blend_weights(dag_path, components)

        for attr in SkinCluster.attributes:
            cmds.setAttr('{0}.{1}'.format(self.node, attr), self.data[attr])

    def set_influence_weights(self, dag_path, components):
        """Sets all the influence weights.

        :param dag_path: MDagPath of the deformed geometry.
        :param components: Component MObject of the deformed components.
        """
        weights = self.__get_current_weights(dag_path, components)
        influence_paths = OpenMaya.MDagPathArray()
        influence_count = self.fn.influenceObjects(influence_paths)
        components_per_influence = weights.length() / influence_count

        # Keep track of which imported influences aren't used
        unused_imports = []

        # Keep track of which existing influences don't get anything imported
        no_match = [influence_paths[ii].partialPathName() for ii in range(influence_paths.length())]

        for imported_influence, imported_weights in self.data['weights'].items():
            for ii in range(influence_paths.length()):
                influence_name = influence_paths[ii].partialPathName()
                influence_without_namespace = shortcuts.remove_namespace_from_name(influence_name)
                if influence_without_namespace == imported_influence:
                    # Store the imported weights into the MDoubleArray
                    for jj in range(components_per_influence):
                        weights.set(imported_weights[jj], jj*influence_count+ii)
                    no_match.remove(influence_name)
                    break
            else:
                unused_imports.append(imported_influence)

        if unused_imports and no_match:
            mapping_dialog = WeightRemapDialog()
            mapping_dialog.set_influences(unused_imports, no_match)
            mapping_dialog.exec_()
            for src, dst in mapping_dialog.mapping.items():
                for ii in range(influence_paths.length()):
                    if influence_paths[ii].partialPathName() == dst:
                        for jj in range(components_per_influence):
                            weights.set(self.data['weights'][src][jj], jj*influence_count+ii)
                        break

        influence_indices = OpenMaya.MIntArray(influence_count)
        for ii in range(influence_count):
            influence_indices.set(ii, ii)
        self.fn.setWeights(dag_path, components, influence_indices, weights, False);

    def set_blend_weights(self, dag_path, components):
        """Set the blendWeights.

        :param dag_path: MDagPath of the deformed geometry.
        :param components: Component MObject of the deformed components.
        """
        blend_weights = OpenMaya.MDoubleArray(len(self.data['blendWeights']))
        for i, w in enumerate(self.data['blendWeights']):
            blend_weights.set(w, i)
        self.fn.setBlendWeights(dag_path, components, blend_weights)


class WeightRemapDialog(MayaQWidgetBaseMixin, QtGui.QDialog):

    def __init__(self, parent=None):
        super(WeightRemapDialog, self).__init__(parent)
        self.setWindowTitle('Remap Weights')
        self.setObjectName('remapWeightsUI')
        self.setModal(True)
        self.resize(600, 400)
        self.mapping = {}

        mainvbox = QtGui.QVBoxLayout(self)

        label = QtGui.QLabel('The following influences have no corresponding influence from the ' \
                             'imported file.  You can either remap the influences or skip them.')
        label.setWordWrap(True)
        mainvbox.addWidget(label)

        hbox = QtGui.QHBoxLayout()
        mainvbox.addLayout(hbox)

        # The existing influences that didn't have weight imported
        vbox = QtGui.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addWidget(QtGui.QLabel('Unmapped influences'))
        self.existing_influences = QtGui.QListWidget()
        vbox.addWidget(self.existing_influences)

        vbox = QtGui.QVBoxLayout()
        hbox.addLayout(vbox)
        vbox.addWidget(QtGui.QLabel('Available imported influences'))
        widget = QtGui.QScrollArea()
        self.imported_influence_layout = QtGui.QVBoxLayout(widget)
        vbox.addWidget(widget)

        hbox = QtGui.QHBoxLayout()
        mainvbox.addLayout(hbox)
        hbox.addStretch()
        btn = QtGui.QPushButton('Ok')
        btn.released.connect(self.accept)
        hbox.addWidget(btn)

    def set_influences(self, imported_influences, existing_influences):
        infs = list(existing_influences)
        infs.sort()
        self.existing_influences.addItems(infs)
        width = 200
        for inf in imported_influences:
            row = QtGui.QHBoxLayout()
            self.imported_influence_layout.addLayout(row)
            label = QtGui.QLabel(inf)
            row.addWidget(label)
            toggle_btn = QtGui.QPushButton('>')
            toggle_btn.setMaximumWidth(30)
            row.addWidget(toggle_btn)
            label = QtGui.QLabel('')
            label.setMaximumWidth(width)
            label.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
            row.addWidget(label)
            toggle_btn.released.connect(partial(self.set_influence_mapping, src=inf, label=label))
        self.imported_influence_layout.addStretch()

    def set_influence_mapping(self, src, label):
        selected_influence = self.existing_influences.selectedItems()
        if not selected_influence:
            return
        dst = selected_influence[0].text()
        label.setText(dst)
        self.mapping[src] = dst
        # Remove the item from the list
        index = self.existing_influences.indexFromItem(selected_influence[0])
        item = self.existing_influences.takeItem(index.row())
        del item

