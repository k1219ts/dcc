
import math
import sys
import maya.OpenMaya as OpenMaya
import maya.OpenMayaMPx as OpenMayaMpx

kPluginNodeTypeName = "dxIk"

kPluginNodeId = OpenMaya.MTypeId(0x00124843)


def slerp(qa, qb, t):
    """Calculates the quaternion slerp between two quaternions.

    From: http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm

    :param qa: Start MQuaternion.
    :param qb: End MQuaternion.
    :param t: Parameter between 0.0 and 1.0
    :return: An MQuaternion interpolated between qa and qb.
    """
    qm = OpenMaya.MQuaternion()

    # Calculate angle between them.
    cos_half_theta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
    # if qa == qb or qa == -qb then theta = 0 and we can return qa
    if abs(cos_half_theta) >= 1.0:
        qm.w = qa.w
        qm.x = qa.x
        qm.y = qa.y
        qm.z = qa.z
        return qa

    # Calculate temporary values
    half_theta = math.acos(cos_half_theta)
    sin_half_theta = math.sqrt(1.0 - cos_half_theta * cos_half_theta)
    # if theta = 180 degrees then result is not fully defined
    # we could rotate around any axis normal to qa or qb
    if math.fabs(sin_half_theta) < 0.001:
        qm.w = (qa.w * 0.5 + qb.w * 0.5)
        qm.x = (qa.x * 0.5 + qb.x * 0.5)
        qm.y = (qa.y * 0.5 + qb.y * 0.5)
        qm.z = (qa.z * 0.5 + qb.z * 0.5)
        return qm

    ratio_a = math.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = math.sin(t * half_theta) / sin_half_theta
    # Calculate quaternion
    qm.w = (qa.w * ratio_a + qb.w * ratio_b)
    qm.x = (qa.x * ratio_a + qb.x * ratio_b)
    qm.y = (qa.y * ratio_a + qb.y * ratio_b)
    qm.z = (qa.z * ratio_a + qb.z * ratio_b)
    return qm

class Pose():
    position = [OpenMaya.MPoint(), OpenMaya.MPoint(), OpenMaya.MPoint()]
    orientation = [OpenMaya.MQuaternion(), OpenMaya.MQuaternion(), OpenMaya.MQuaternion()]


class DxIk(OpenMayaMpx.MPxNode):
    iRootMatrix = OpenMaya.MObject()
    iGoalMatrix = OpenMaya.MObject()
    iPoleMatrix = OpenMaya.MObject()
    iFkMatrix = OpenMaya.MObject()

    iStretchBlend = OpenMaya.MObject()
    iFkIkBlend = OpenMaya.MObject()
    iPinBlend = OpenMaya.MObject()
    iReverseBlend = OpenMaya.MObject()
    iOrientTipBlend = OpenMaya.MObject()
    iFlipOrientation = OpenMaya.MObject()

    iUpperLength = OpenMaya.MObject()
    iLowerLength = OpenMaya.MObject()
    iUpperLengthBoost = OpenMaya.MObject()
    iLowerLengthBoost = OpenMaya.MObject()
    iLengthBoost = OpenMaya.MObject()
    iSoftness = OpenMaya.MObject()
    iTwist = OpenMaya.MObject()

    oOutTranslate = OpenMaya.MObject()
    oOutRotateX = OpenMaya.MObject()
    oOutRotateY = OpenMaya.MObject()
    oOutRotateZ = OpenMaya.MObject()
    oOutRotate = OpenMaya.MObject()


    def __init__(self):
        OpenMayaMpx.MPxNode.__init__(self)

    def matrix_from_three_vectors(self, a, b, c):
        mat_data = [a.x, a.y, a.z, 0.0,
                    b.x, b.y, b.z, 0.0,
                    c.x, c.y, c.z, 0.0,
                    0.0, 0.0, 0.0, 1.0]
        mat_data_matrix = OpenMaya.MMatrix()
        OpenMaya.MScriptUtil.createMatrixFromList(mat_data, mat_data_matrix)
        return mat_data_matrix

    def matrix_from_two_vectors(self, aim, up, flip=False):
        d_flip = 1.0
        if flip: d_flip *= -1

        cross = up ^ aim
        return self.matrix_from_three_vectors(aim * d_flip, cross * d_flip, up * d_flip)


    def calculate_orientations(self, pose, pole_point, flip):
        upper_vector = pose.position[1] - pose.position[0]
        lower_vector = pose.position[2] - pose.position[1]
        goal_vector = pose.position[2] - pose.position[0]

        upper_vector.normalize()
        lower_vector.normalize()

        pole_vector = pole_point - pose.position[0]
        new_pole_point = pose.position[0] + goal_vector * ((pole_vector * goal_vector) / (goal_vector * goal_vector))
        pole_vector = pole_point - new_pole_point
        pole_vector.normalize()

        side_vector = goal_vector ^ pole_vector

        pose.orientation[0] = OpenMaya.MTransformationMatrix(self.matrix_from_two_vectors(upper_vector, side_vector, flip)).rotation()
        pose.orientation[1] = OpenMaya.MTransformationMatrix(self.matrix_from_two_vectors(lower_vector, side_vector, flip)).rotation()


    def spin_vector_around_axis(self, vec, angle, axis):
        projected = axis * (vec * axis)
        return (vec - projected) * math.cos(angle) + (axis ^ vec) * math.sin(angle) + projected

    def clamp(self, num, min, max):
        num = min if num < min else max if num > max else num
        return num

    def lerp(self, start, end, percent):
        return start + (start - end) * percent

    def compute(self, plug, data):
        node = self.thisMObject()
        MIN_SOFTNESS = 0.00000001

        plug_check = plug == self.oOutTranslate or plug == self.oOutRotate\
            or plug == self.oOutRotateX or plug == self.oOutRotateY or plug == self.oOutRotateZ

        if not plug_check:
            return OpenMaya.kUnknownParameter

        rootMatrix = data.inputValue(self.iRootMatrix).asMatrix()
        goalMatrix = data.inputValue(self.iGoalMatrix).asMatrix()
        poleMatrix = data.inputValue(self.iPoleMatrix).asMatrix()
        stretchBlend = data.inputValue(self.iStretchBlend).asDouble()
        fkIkBlend = data.inputValue(self.iFkIkBlend).asDouble()
        pinBlend = data.inputValue(self.iPinBlend).asDouble()
        reverseBlend = data.inputValue(self.iReverseBlend).asDouble()
        orientTipBlend = data.inputValue(self.iOrientTipBlend).asDouble()
        flipOrientation = data.inputValue(self.iFlipOrientation).asDouble()
        upperLength = data.inputValue(self.iUpperLength).asDouble()
        lowerLength = data.inputValue(self.iLowerLength).asDouble()
        upperLengthBoost = data.inputValue(self.iUpperLengthBoost).asDouble()
        lowerLengthBoost = data.inputValue(self.iLowerLengthBoost).asDouble()
        lengthBoost = data.inputValue(self.iLengthBoost).asDouble()
        softness = data.inputValue(self.iSoftness).asDouble()
        twist = data.inputValue(self.iTwist).asDouble()

        scale_compensate = OpenMaya.MVector(OpenMaya.MScriptUtil.getDoubleArrayItem(rootMatrix[0], 0),
                                            OpenMaya.MScriptUtil.getDoubleArrayItem(rootMatrix[0], 1),
                                            OpenMaya.MScriptUtil.getDoubleArrayItem(rootMatrix[0], 2)).length()
        inverse_scale = 1.0 / scale_compensate

        if softness < MIN_SOFTNESS: softness = MIN_SOFTNESS

        lengthBoost *= 0.01
        upperLengthBoost *= 0.01
        lowerLengthBoost *= 0.01

        twist = math.radians(twist)

        realUpperLength = upperLength * upperLengthBoost * lengthBoost
        realLowerLength = lowerLength * lowerLengthBoost * lengthBoost

        ik_pose = Pose()
        fk_pose = Pose()

        rootMatrixInverse = rootMatrix.inverse()
        rootPos = OpenMaya.MTransformationMatrix(rootMatrix).translation(OpenMaya.MSpace.kTransform)
        mt_goalMatrix = OpenMaya.MTransformationMatrix(goalMatrix)
        goalPos = mt_goalMatrix.translation(OpenMaya.MSpace.kTransform)
        mt_poleMatrix = OpenMaya.MTransformationMatrix(poleMatrix)
        polePos = mt_poleMatrix.translation(OpenMaya.MSpace.kTransform)

        if fkIkBlend < 1.0:
            h_fkMatrix = data.inputArrayValue(self.iFkMatrix)
            scale_mat = OpenMaya.MMatrix()
            for index in range(3):
                OpenMaya.MScriptUtil.setDoubleArray(scale_mat[index], index , inverse_scale)

            if h_fkMatrix.elementCount() > 2:
                for index in range(3):
                    h_fkMatrix.jumpToArryElement(index)
                    mt_mat = OpenMaya.MTransformationMatrix(h_fkMatrix.inputValue().asMatrix())
                    fk_pose.orientation[index] = mt_mat.rotation()
                    fk_pose.position[index] = mt_mat.translation(OpenMaya.MSpace.kTransform)
                    fk_pose.position[index] -= rootPos
                    fk_pose.position[index] *= scale_mat
                    fk_pose.position[index] += rootPos

        goal_vector = goalPos - rootPos
        chainLength = realUpperLength + realLowerLength
        distance = goal_vector.length() * inverse_scale
        soft_distance = chainLength - softness
        adjusted_distance = distance
        scale = 1.0

        goal_vector.normalize()

        ik_pose.position[0] = rootPos

        if fkIkBlend != 0.0:
            if distance > soft_distance and pinBlend != 1.0:
                k = softness * (1.0 - math.exp(-1.0 * (distance-soft_distance)/softness)) + soft_distance
                smartRatio = k / chainLength
                lenRatio = distance / chainLength
                adjusted_distance = distance / lenRatio * smartRatio

                scale = (distance / adjusted_distance - 1.0) * stretchBlend + 1.0
                realUpperLength *= scale
                realLowerLength *= scale
                adjusted_distance *= scale

        pole_vector = polePos - rootPos
        pole_point = rootPos + goal_vector * ((pole_vector * goal_vector) / (goal_vector * goal_vector))
        pole_vector = polePos - pole_point

        pole_vector = self.spin_vector_around_axis(pole_vector, twist, goal_vector)
        polePos = pole_point + pole_vector

        pole_vector.normalize()
        side_vector = pole_vector ^ goal_vector
        side_vector.normalize()

        root_cosine = ((realUpperLength * realUpperLength) + (adjusted_distance * adjusted_distance) -\
                       (realLowerLength * realLowerLength)) / (2.0 * realUpperLength * adjusted_distance)

        root_cosine_clamped = self.clamp(root_cosine, -1.0, 1.0)
        upper_angle = math.acos(root_cosine_clamped)

        if upper_angle < 0.0001:
            upper_bone_vector = goal_vector
        else:
            goal_delta_quat = OpenMaya.MQuaternion()
            goal_delta_quat.setAxisAngle(side_vector, upper_angle)
            r_quat = goal_delta_quat * OpenMaya.MQuaternion(goal_vector.x, goal_vector.y, goal_vector.z, 0.0) * goal_delta_quat.inverse()
            upper_bone_vector = OpenMaya.MVector(r_quat.x, r_quat.y, r_quat.z)

        ik_pose.position[1] = ik_pose.position[0] + (upper_bone_vector * realUpperLength)

        lower_bone_vector = (ik_pose.position[0] + (goal_vector * adjusted_distance)) - ik_pose.position[1]
        ik_pose.position[2] = ik_pose.position[1] + lower_bone_vector
        lower_bone_vector.normalize()

        if pinBlend > 0.0:
            if pinBlend == 1.0:
                ik_pose.position[1] = polePos
                ik_pose.position[2] = goalPos
            else:
                ik_pose.position[1] = self.lerp(ik_pose.position[1], polePos, pinBlend)
                ik_pose.position[2] = self.lerp(ik_pose.position[2], polePos, pinBlend)

        reverseBlend *= 1.0 - pinBlend

        if reverseBlend > 0.0:
            reflect_angle = math.acos(upper_bone_vector * goal_vector)

            if not math.isnan(reflect_angle):
                reflected = self.spin_vector_around_axis(upper_bone_vector,
                                                         reflect_angle * 2.0,
                                                         side_vector)
                reflected_pos = rootPos + (reflected * realUpperLength)
                if reverseBlend == 1.0:
                    ik_pose.position[1] = reflected_pos
                else:
                    ik_pose.position[1] = self.lerp(ik_pose.position[1], reflected_pos, reverseBlend)

        self.calculate_orientations(ik_pose, polePos, flipOrientation)

        if orientTipBlend == 0.0:
            ik_pose.orientation[2] = ik_pose.orientation[1]
        else:
            goal_quat = mt_goalMatrix.rotation()
            if orientTipBlend == 1.0:
                ik_pose.orientation[2] = goal_quat
            else:
                ik_pose.orientation[2] = slerp(ik_pose.orientation[1], goal_quat, orientTipBlend)

        final_pose = Pose()

        if fkIkBlend == 0.0:
            final_pose = fk_pose
        elif fkIkBlend == 1.0:
            final_pose = ik_pose
        else:
            for index in range(3):
                final_pose.position[index] = self.lerp(fk_pose.position[index], ik_pose.position[index], fkIkBlend)
                final_pose.orientation[index] = slerp(fk_pose.orientation[index], ik_pose.orientation[index], fkIkBlend)

        h_outTranslate = data.outputArrayValue(self.oOutTranslate)
        h_outRotate = data.outputArrayValue(self.oOutRotate)

        b_outTranslate = h_outTranslate.builder()
        b_outRotate = h_outRotate.builder()

        mt_previous = OpenMaya.MTransformationMatrix(rootMatrix)

        for index in range(3):
            mt_out = OpenMaya.MTransformationMatrix()
            mt_out.setTranslation(final_pose.position[index], OpenMaya.MSpace.kTransform)
            rot_quat = final_pose.orientation[index]
            mt_out.setRotationQuaternion(rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w)
            mt_final = mt_out.asMatrix() * mt_previous.asMatrixInverse()
            mt_final_mtx = OpenMaya.MTransformationMatrix(mt_final)
            mt_previous = mt_out

            b_outTranslate.addElement(index).setMFloatVector(OpenMaya.MFloatVector(mt_final_mtx.getTranslation(OpenMaya.MSpace.kTransform)))
            rotation = mt_final_mtx.eulerRotation()
            b_outRotate.addElement(index).set3Double(rotation.x, rotation.y, rotation.z)

        h_outTranslate.setAllClean()
        h_outRotate.setAllClean()


def nodeCreator():
    nodePtr = OpenMayaMpx.asMPxPtr(DxIk())
    return nodePtr

def nodeInitializer():
    uAttr = OpenMaya.MFnUnitAttribute()
    nAttr = OpenMaya.MFnNumericAttribute()
    mAttr = OpenMaya.MFnMatrixAttribute()

    identity = OpenMaya.MMatrix()
    identity.setToIdentity()

    DxIk.iStretchBlend = nAttr.create("stretchBlend", "stretchBlend", OpenMaya.MFnNumericData.kDouble, 1.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)

    DxIk.iFkIkBlend = nAttr.create("fkIkBlend", "fkIkBlend", OpenMaya.MFnNumericData.kDouble, 1.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)

    DxIk.iPinBlend = nAttr.create("pinBlend", "pinBlend", OpenMaya.MFnNumericData.kDouble, 0.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)

    DxIk.iReverseBlend = nAttr.create("reverseBlend", "reverseBlend", OpenMaya.MFnNumericData.kDouble, 0.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)

    DxIk.iOrientTipBlend = nAttr.create("orientTipBlend", "orientTipBlend", OpenMaya.MFnNumericData.kDouble, 1.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(1.0)

    DxIk.iFlipOrientation = nAttr.create("flipOrientation", "flipOrientation", OpenMaya.MFnNumericData.kBoolean, False)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)

    DxIk.iUpperLength = nAttr.create("upperLength", "upperLength", OpenMaya.MFnNumericData.kDouble, -1.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)

    DxIk.iLowerLength = nAttr.create("lowerLength", "lowerLength", OpenMaya.MFnNumericData.kDouble, -1.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)

    DxIk.iUpperLengthBoost = nAttr.create("upperLengthBoost", "upperLengthBoost", OpenMaya.MFnNumericData.kDouble, 100.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)

    DxIk.iLowerLengthBoost = nAttr.create("lowerLengthBoost", "lowerLengthBoost", OpenMaya.MFnNumericData.kDouble, 100.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)

    DxIk.iLengthBoost = nAttr.create("lengthBoost", "lengthBoost", OpenMaya.MFnNumericData.kDouble, 100.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)

    DxIk.iSoftness = nAttr.create("softness", "softness", OpenMaya.MFnNumericData.kDouble, 0.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)
    nAttr.setMin(0.0)
    nAttr.setMax(10.0)

    DxIk.iTwist = nAttr.create("twist", "twist", OpenMaya.MFnNumericData.kDouble, 0.0)
    nAttr.setStorable(True)
    nAttr.setKeyable(True)
    nAttr.setReadable(True)
    nAttr.setWritable(True)

    DxIk.iRootMatrix = mAttr.create("rootMatrix", "rootMatrix")
    mAttr.setDefault(identity)
    mAttr.setStorable(True)
    mAttr.setKeyable(False)
    mAttr.setCached(True)
    mAttr.setReadable(False)
    mAttr.setWritable(True)

    DxIk.iGoalMatrix = mAttr.create("goalMatrix", "goalMatrix")
    mAttr.setDefault(identity)
    mAttr.setStorable(True)
    mAttr.setKeyable(False)
    mAttr.setCached(True)
    mAttr.setReadable(False)
    mAttr.setWritable(True)

    DxIk.iPoleMatrix = mAttr.create("poleMatrix", "poleMatrix")
    mAttr.setDefault(identity)
    mAttr.setStorable(True)
    mAttr.setKeyable(False)
    mAttr.setCached(True)
    mAttr.setReadable(False)
    mAttr.setWritable(True)

    DxIk.iFkMatrix = mAttr.create("fkMatrix", "fkMatrix")
    mAttr.setDefault(identity)
    mAttr.setStorable(True)
    mAttr.setKeyable(False)
    mAttr.setCached(True)
    mAttr.setReadable(False)
    mAttr.setWritable(True)
    mAttr.setArray(True)

    DxIk.oOutTranslate = nAttr.createPoint("outTranslate", "outTranslate")
    nAttr.setStorable(True)
    nAttr.setKeyable(False)
    nAttr.setWritable(False)
    nAttr.setReadable(True)
    nAttr.setArray(True)
    nAttr.setUsesArrayDataBuilder(True)

    DxIk.oOutRotateX = uAttr.create("outRotateX", "outRotateX", OpenMaya.MFnUnitAttribute.kAngle, 0.0)
    uAttr.setStorable(False)
    uAttr.setWritable(False)

    DxIk.oOutRotateY = uAttr.create("outRotateY", "outRotateY", OpenMaya.MFnUnitAttribute.kAngle, 0.0)
    uAttr.setStorable(False)
    uAttr.setWritable(False)

    DxIk.oOutRotateZ = uAttr.create("outRotateZ", "outRotateZ", OpenMaya.MFnUnitAttribute.kAngle, 0.0)
    uAttr.setStorable(False)
    uAttr.setWritable(False)

    DxIk.oOutRotate = nAttr.create("outRotate", "outRotate", DxIk.oOutRotateX, DxIk.oOutRotateY, DxIk.oOutRotateZ)
    nAttr.setReadable(True)
    nAttr.setWritable(False)
    nAttr.setArray(True)
    nAttr.setStorable(False)
    nAttr.setUsesArrayDataBuilder(True)

    all_inputs = [DxIk.iStretchBlend,
                  DxIk.iFkIkBlend,
                  DxIk.iPinBlend,
                  DxIk.iReverseBlend,
                  DxIk.iOrientTipBlend,
                  DxIk.iFlipOrientation,
                  DxIk.iUpperLength,
                  DxIk.iLowerLength,
                  DxIk.iUpperLengthBoost,
                  DxIk.iLowerLengthBoost,
                  DxIk.iLengthBoost,
                  DxIk.iSoftness,
                  DxIk.iTwist,
                  DxIk.iRootMatrix,
                  DxIk.iGoalMatrix,
                  DxIk.iPoleMatrix,
                  DxIk.iFkMatrix]

    all_outputs = [DxIk.oOutTranslate,
                   DxIk.oOutRotate]

    DxIk.addAttribute(DxIk.oOutTranslate)
    DxIk.addAttribute(DxIk.oOutRotate)

    for inputAttr in all_inputs:
        DxIk.addAttribute(inputAttr)
        for outputAttr in all_outputs:
            DxIk.attributeAffects(inputAttr, outputAttr)


def initializePlugin(mobject):
    mplugin = OpenMayaMpx.MFnPlugin(mobject, "gyeongheon.jeong", "1.0")
    try:
        mplugin.registerNode(kPluginNodeTypeName, kPluginNodeId, nodeCreator, nodeInitializer)
    except:
        sys.stderr.write("Failed to register node: {0}\n".format(kPluginNodeTypeName))


def uninitializePlugin(mobject):
    mplugin = OpenMayaMpx.MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(kPluginNodeId)
    except:
        sys.stderr.write("Failed to unregister node: {0}\n".format(kPluginNodeTypeName))
