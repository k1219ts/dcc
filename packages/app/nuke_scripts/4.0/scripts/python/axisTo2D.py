################
'''

	sb_Axis_to_2D
	Simon Bjork
	November 2016

	Generate 2d keyframes from selected Axis nodes.

	---------------------

	Updated some of the nukescripts.snap3d functions to accept a frame and format argument.
	Ivan Busquets posted the code to give the frame argument to CameraProjectionMatrix()
	http://forums.thefoundry.co.uk/phpBB2/viewtopic.php?p=26502&sid=90d2f80543be7b2bf2decbeb556eaea7

	---------------------

	Note:

	As the main part of the code was written quite a few years ago, there's possibly a better way to do this now with the OutputContext (Nuke 6.5+).

'''

################

import nuke
import _nukemath
import nukescripts
import math


################

def shuffleWorldMatrix(node, frame):
    matrixList = node["world_matrix"].valueAt(frame)
    # Reorder list and put it into a matrix.
    order = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    matrix = _nukemath.Matrix4()
    for i in range(16):
        matrix[i] = matrixList[order[i]]
    return matrix


# Updated version.
def fixedCameraProjectionMatrix(cameraNode, format, frame):
    # Calculate the projection matrix for the camera, based on it's knob values.

    shuffledWorldMatrix = shuffleWorldMatrix(cameraNode, frame)
    camTransform = shuffledWorldMatrix.inverse()

    # Matrix to take the camera projection knobs into account
    roll = float(cameraNode['winroll'].getValueAt(frame))
    scale_x, scale_y = [float(v) for v in cameraNode['win_scale'].getValueAt(frame)]
    translate_x, translate_y = [float(v) for v in cameraNode['win_translate'].getValueAt(frame)]
    m = _nukemath.Matrix4()
    m.makeIdentity()
    m.rotateZ(math.radians(roll))
    m.scale(1.0 / scale_x, 1.0 / scale_y, 1.0)
    m.translate(-translate_x, -translate_y, 0.0)

    # Projection matrix based on the focal length, aperture and clipping planes of the camera
    focal_length = float(cameraNode['focal'].getValueAt(frame))
    h_aperture = float(cameraNode['haperture'].getValueAt(frame))
    near = float(cameraNode['near'].getValueAt(frame))
    far = float(cameraNode['far'].getValueAt(frame))
    projection_mode = int(cameraNode['projection_mode'].getValueAt(frame))
    p = _nukemath.Matrix4()
    p.projection(focal_length / h_aperture, near, far, projection_mode == 0)

    # Matrix to translate the projected points into normalised pixel coords.
    imageAspect = float(format.height()) / float(format.width())
    t = _nukemath.Matrix4()
    t.makeIdentity()
    t.translate(1.0, 1.0 - (1.0 - imageAspect / float(format.pixelAspect())), 0.0)

    # Matrix to scale normalised pixel coords into actual pixel coords.
    x_scale = float(format.width()) / 2.0
    y_scale = x_scale * format.pixelAspect()
    s = _nukemath.Matrix4()
    s.makeIdentity()
    s.scale(x_scale, y_scale, 1.0)

    # The projection matrix transforms points into camera coords, modifies based
    # on the camera knob values, projects points into clip coords, translates the
    # clip coords so that they lie in the range 0,0 - 2,2 instead of -1,-1 - 1,1,
    # then scales the clip coords to proper pixel coords.
    return s * t * p * m * camTransform


# Updated version.
def fixedProjectPoints(camera=None, points=None, format=None, frame=None):
    camNode = None
    if isinstance(camera, nuke.Node):
        camNode = camera
    elif isinstance(camera, str):
        camNode = nuke.toNode(camera)
    else:
        raise ValueError("Argument camera must be a node or the name of a node.")

    camMatrix = fixedCameraProjectionMatrix(camNode, format, frame)

    if camMatrix == None:
        raise RuntimeError("snap3d.cameraProjectionMatrix() returned None for camera.")

    if not (isinstance(points, list) or isinstance(points, tuple)):
        raise ValueError("Argument points must be a list or tuple.")

    for point in points:
        # Would be nice to not do this for every item but since lists/tuples can
        # containg anything...
        if isinstance(point, nuke.math.Vector3):
            pt = point
        elif isinstance(point, list) or isinstance(point, tuple):
            pt = nuke.math.Vector3(point[0], point[1], point[2])
        else:
            raise ValueError("All items in points must be nuke.math.Vector3 or list/tuple of 3 floats.")

        tPos = camMatrix * nuke.math.Vector4(pt.x, pt.y, pt.z, 1.0)
        yield nuke.math.Vector2(tPos.x / tPos.w, tPos.y / tPos.w)


# Updated version.
def fixedProjectPoint(camera=None, point=None, format=None, frame=None):
    return fixedProjectPoints(camera, (point,), format, frame).next()


def sb_axisTo2D():
    axis_nodes = nuke.selectedNodes("Axis2")

    if not axis_nodes:
        nuke.message("No Axis selected.")
        return

    nukescripts.clear_selection_recursive()

    p = nuke.Panel("Axis to 2D")
    p.addEnumerationPulldown('Camera', " ".join([x.name() for x in nuke.allNodes("Camera2")]))
    p.addSingleLineInput('Start', int(nuke.root()["first_frame"].value()))
    p.addSingleLineInput('End', int(nuke.root()["last_frame"].value()))
    result = p.show()

    if not result:
        print("Cancelled by user.")
        return

    cam = nuke.toNode(p.value("Camera"))
    ff = int(p.value('Start'))
    lf = int(p.value('End'))

    format = nuke.root().format()

    noop = nuke.createNode("NoOp", inpanel=False)
    noop.setName("Axis_2D")
    noop.setInput(0, None)
    noop["tile_color"].setValue(2623383809)

    for axis in axis_nodes:

        name = axis.name()

        xy_knob = nuke.XY_Knob(name.lower(), name)
        noop.addKnob(xy_knob)
        xy_knob.setAnimated()

        # Setup progress bar.
        task = nuke.ProgressTask('Generating...')
        progressCalc = 100.0 / float(lf - ff)
        counter = 0

        for i in range(ff, lf + 1):

            if task.isCancelled():
                nuke.executeInMainThread(nuke.message, args=('Aborted by user.',))

            # World position of Axis.
            m = axis["world_matrix"].valueAt(i)
            wp = (m[3], m[7], m[11])

            try:
                # Convert 3d to 2d.
                pos2d = fixedProjectPoint(cam, wp, format, i)
                xy_knob.setValueAt(pos2d[0], i, 0)
                xy_knob.setValueAt(pos2d[1], i, 1)

            except Exception as e:
                print(e)

            task.setProgress(int(counter * progressCalc))
            task.setMessage(name)
            counter += 1

        # Delete progress bar.
        del task

        # nuke.zoom(1.6817928552627563, (noop.xpos(), noop.ypos()))
