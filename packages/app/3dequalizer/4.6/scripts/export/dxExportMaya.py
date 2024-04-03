import os
import tde4
from vl_sdv import *
import __builtin__ as builtin
import dxUIcommon


class dxExportMel:
    def __init__(self, requester):
        self.req = requester
        self.overscanList = ['1.08', '1.1', '1.15', '1.2', 'custom']
        self.user = os.getenv('USER')
        self.cameraList = tde4.getCameraList()

        pgl	= tde4.getPGroupList()
        for pg in pgl:
            if tde4.getPGroupType(pg)=="CAMERA":	self.campg = pg

        self.projectPath = tde4.getProjectPath()
        if self.projectPath != None:
            self.file_name = self.projectPath.replace('.3de','.mel')

        # unit_scales = {1 : 1.0,2 : 0.01,  3 : 10.0, 4 : 0.393701, 5 : 0.0328084, 6 : 0.0109361}
        # unit_scale_factor = unit_scales[tde4.getWidgetValue(req,"units")]
        # self.unit_scale_factor = 1.0
        self.windowTitle = ''

        if not os.environ.has_key('show'):
            for envKey in ['show', 'seq','shot','platetype']:
                os.environ[envKey] = ''

    def convertToAngles(self, r3d):
        rot	= rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)
        rx	= (rot[0]*180.0)/3.141592654
        ry	= (rot[1]*180.0)/3.141592654
        rz	= (rot[2]*180.0)/3.141592654
        return(rx,ry,rz)

    def convertZup(self, p3d, yup, scale=1.0):
        if yup==1:
            return([p3d[0]*scale,p3d[1]*scale,p3d[2]*scale])
        else:
            return([p3d[0]*scale,-p3d[2]*scale,p3d[1]*scale])

    def angleMod360(self, d0, d):
        dd	= d-d0
        if dd>180.0:
            d	= self.angleMod360(d0,d-360.0)
        else:
            if dd<-180.0:
                d	= self.angleMod360(d0,d+360.0)
        return d

    def validName(self, name):
        name	= name.replace(' ','_')
        name	= name.replace('-','_')
        name	= name.replace('\n','')
        name	= name.replace('\r','')
        return name

    def prepareImagePath(self, path, startframe):
        path	= path.replace('\\','/')
        i	= 0
        n	= 0
        i0	= -1
        while(i<len(path)):
            if path[i]=='#': n += 1
            if n==1: i0 = i
            i	+= 1
        if i0!=-1:
            fstring		= '%%s%%0%dd%%s'%(n)
            path2		= fstring%(path[0:i0],startframe,path[i0+n:len(path)])
            path		= path2
        return path

    def doIt(self):
        camera_selection = tde4.getWidgetValue(self.req, 'camera_selection')
        # model_selection = tde4.getWidgetValue(self.req ,'model_selection')
        # export_material = tde4.getWidgetValue(self.req,'export_texture')

        if camera_selection == 1:
    		self.cameraList = [tde4.getCurrentCamera()]
    	elif camera_selection == 2:
    		self.cameraList = tde4.getCameraList(1)
    	elif camera_selection == 3:
    		self.cameraList = []
    		tcl =  tde4.getCameraList()
    		for c in tcl:
    			if tde4.getCameraType(c) == 'SEQUENCE':
    				self.cameraList.append(c)
    	elif camera_selection == 4:
    		self.cameraList = []
    		tcl =  tde4.getCameraList()
    		for c in tcl:
    			if tde4.getCameraType(c) == 'REF_FRAME':
    				self.cameraList.append(c)

    	current_camera = tde4.getCurrentCamera()
    	if builtin.len(self.cameraList) > 0:
    		current_camera = self.cameraList[0]
    	for c in self.cameraList:
    		if c == tde4.getCurrentCamera():
    			current_camera = c

        stereo = tde4.getWidgetValue(self.req, 'stereo')
        isOverscan = tde4.getWidgetValue(self.req, 'overscan')
        if isOverscan:
            overscan = dxUIcommon.setOverscanWidget(self.req, current_camera).getOverscanValue()
        else:
            overscan = 1.0

        yup	= 1
        path = tde4.getWidgetValue(self.req, 'file_browser')
        frame0 = int(tde4.getWidgetValue(self.req, 'start_frame'))
        frame0 -= 1
        hide_ref = tde4.getWidgetValue(self.req,'hide_ref_frames')
        export_model = tde4.getWidgetValue(self.req,'export_3dmodel')

        if path!=None:
            if not path.endswith('.mel'): path = path+'.mel'
            f	= open(path,'w')
            if not f.closed:
                f.write('//\n')
                f.write('// Maya/MEL export data written by %s\n'%tde4.get3DEVersion())
                f.write('//\n')
                f.write('// All lengths are in centimeter, all angles are in degree.\n')
                f.write('//\n\n')

                f.write('string $sceneGroupName = `createNode dxCamera`;\n')

                index = 1
                for cam in self.cameraList:
                    camType = tde4.getCameraType(cam)
                    noframes = tde4.getCameraNoFrames(cam)
                    lens = tde4.getCameraLens(cam)
                    if lens!=None:
                        name = self.validName(tde4.getCameraName(cam))
                        if name.count('_') >= 3:
                            name = name.split('_')
                            name = '%s_cam' % '_'.join(name[:-1])
                        else:
                            name = name.replace('.', '_')

                        index += 1
                        fback_w = tde4.getLensFBackWidth(lens)
                        fback_h = tde4.getLensFBackHeight(lens)
                        p_aspect = tde4.getLensPixelAspect(lens)
                        focal = tde4.getCameraFocalLength(cam,1)
                        lco_x = tde4.getLensLensCenterX(lens)
                        lco_y = tde4.getLensLensCenterY(lens)

                        # convert filmback to inch...
                        fback_w = fback_w/2.54
                        fback_h = fback_h/2.54
                        lco_x = -lco_x/2.54
                        lco_y = -lco_y/2.54

                        # convert focal length to mm...
                        focal = focal*10.0

                        # set render global.
                        image_w = tde4.getCameraImageWidth(cam)
                        image_h = tde4.getCameraImageHeight(cam)
                        f.write('setAttr \"defaultResolution.width\" %s;\n'%image_w)
                        f.write('setAttr \"defaultResolution.height\" %s;\n'%image_h)
                        f.write('setAttr \"defaultResolution.deviceAspectRatio\" %.8f;\n'%(float(image_w)/float(image_h)))
                        f.write('setAttr \"defaultRenderGlobals.animation\" 1;\n')
                        f.write('setAttr \"defaultRenderGlobals.extensionPadding\" 4;\n')
                        f.write('setAttr \"defaultRenderGlobals.putFrameBeforeExt\" 1;\n')
                        f.write('setAttr \"defaultRenderGlobals.startFrame\" %d;\n'%(1+frame0))
                        f.write('setAttr \"defaultRenderGlobals.endFrame\" %d;\n'%(noframes+frame0))

                        # create camera...
                        f.write('\n')
                        f.write('// create camera %s...\n'%name)
                        f.write('createNode \"camera\" - n \"%sShape\";\n' % name)
                        f.write('string $cameraNodes[] = `ls \"%s*\"`;\n' % name)
                        f.write('camera -e -hfa %.15f  -vfa %.15f -fl %.15f -ncp 0.1 -fcp 100000 -shutterAngle 180 -ff \"horizontal\" $cameraNodes[0];\n'%(fback_w*overscan, fback_h*overscan, focal))
                        f.write('string $cameraTransform = $cameraNodes[0];\n')
                        f.write('string $cameraShape = $cameraNodes[1];\n')
                        f.write('xform -zeroTransformPivots -rotateOrder zxy $cameraTransform;\n')
                        f.write('setAttr ($cameraShape+\".horizontalFilmOffset\") %.15f;\n'%lco_x);
                        f.write('setAttr ($cameraShape+\".verticalFilmOffset\") %.15f;\n'%lco_y);
                        p3d = tde4.getPGroupPosition3D(self.campg,cam,1)
                        p3d = self.convertZup(p3d,yup)
                        f.write('xform -translation %.15f %.15f %.15f $cameraTransform;\n'%(p3d[0],p3d[1],p3d[2]))
                        r3d = tde4.getPGroupRotation3D(self.campg,cam,1)
                        rot = self.convertToAngles(r3d)
                        f.write('xform -rotation %.15f %.15f %.15f $cameraTransform;\n'%rot)
                        f.write('xform -scale 1 1 1 $cameraTransform;\n')

                        # image plane...
                        f.write('\n')
                        f.write('// create image plane...\n')
                        f.write('string $imagePlane = `createNode imagePlane`;\n')
                        f.write('cameraImagePlaneUpdate ($cameraShape, $imagePlane);\n')
                        f.write('setAttr ($imagePlane + \".offsetX\") %.15f;\n'%lco_x)
                        f.write('setAttr ($imagePlane + \".offsetY\") %.15f;\n'%lco_y)

                        if camType=='SEQUENCE': f.write('setAttr ($imagePlane+\".useFrameExtension\") 1;\n')
                        else: f.write('setAttr ($imagePlane+\".useFrameExtension\") 0;\n')

                        # f.write('expression -n \"frame_ext_expression\" -s ($imagePlane+\".frameExtension=frame\");\n')

                        tde4.setCameraProxyFootage(cam, 3)
                        if tde4.getCameraPath(cam) == '':
                            tde4.setCameraProxyFootage(cam, 0)

                        path = tde4.getCameraPath(cam)
                        tde4.setCameraProxyFootage(cam, 0)
                        sattr = tde4.getCameraSequenceAttr(cam)
                        path = self.prepareImagePath(path,sattr[0])
                        f.write('setAttr ($imagePlane + \".imageName\") -type \"string\" \"%s\";\n'%(path))
                        f.write('setAttr ($imagePlane + \".fit\") 4;\n')
                        f.write('setAttr ($imagePlane + \".displayOnlyIfCurrent\") 1;\n')
                        f.write('setAttr ($imagePlane  + \".depth\") (9000/2);\n')

                        # parent camera to scene group...
                        f.write('\n')
                        f.write('// parent camera to scene group...\n')
                        f.write('parent $cameraTransform $sceneGroupName;\n')

                        if camType=='REF_FRAME' and hide_ref:
                            f.write('setAttr ($cameraTransform +\".visibility\") 0;\n')

                        # animate camera...
                        if camType!='REF_FRAME':
                            f.write('\n')
                            f.write('// animating camera %s...\n'%name)
                            f.write('playbackOptions -ast %d -aet %d -min %d -max %d;\n'%(1+frame0, noframes+frame0, 1+frame0, noframes+frame0))
                            f.write('currentTime %d;\n'%(1+frame0))
                            f.write('\n')

                        frame = 1
                        while frame<=noframes:
                            # rot/pos...
                            p3d = tde4.getPGroupPosition3D(self.campg,cam,frame)
                            p3d = self.convertZup(p3d,yup)
                            r3d = tde4.getPGroupRotation3D(self.campg,cam,frame)
                            rot = self.convertToAngles(r3d)
                            if frame>1:
                                rot = [ self.angleMod360(rot0[0],rot[0]), self.angleMod360(rot0[1],rot[1]), self.angleMod360(rot0[2],rot[2]) ]
                            rot0 = rot
                            f.write('setKeyframe -at translateX -t %d -v %.15f $cameraTransform; '%(frame+frame0,p3d[0]))
                            f.write('setKeyframe -at translateY -t %d -v %.15f $cameraTransform; '%(frame+frame0,p3d[1]))
                            f.write('setKeyframe -at translateZ -t %d -v %.15f $cameraTransform; '%(frame+frame0,p3d[2]))
                            f.write('setKeyframe -at rotateX -t %d -v %.15f $cameraTransform; '%(frame+frame0,rot[0]))
                            f.write('setKeyframe -at rotateY -t %d -v %.15f $cameraTransform; '%(frame+frame0,rot[1]))
                            f.write('setKeyframe -at rotateZ -t %d -v %.15f $cameraTransform; '%(frame+frame0,rot[2]))

                            # focal length...
                            focal = tde4.getCameraFocalLength(cam,frame)
                            focal = focal*10.0
                            f.write('setKeyframe -at focalLength -t %d -v %.15f $cameraShape;\n'%(frame+frame0,focal))

                            frame += 1

                # write scene info...
                f.write('\n')
                f.write('// write scene info...\n')
                f.write('fileInfo \"3deProject\" \"%s\";\n'%self.projectPath)
                if isOverscan:
                    f.write('fileInfo \"overscan\" \"true\";\n')
                else:
                    f.write('fileInfo \"overscan\" \"false\";\n')
                f.write('fileInfo \"overscan_value\" \"%s\";\n' % str(overscan))

                f.write('fileInfo \"resWidth\" \"%s\";\n'%image_w)
                f.write('fileInfo \"resHeight\" \"%s\";\n'%image_h)
                f.write('fileInfo \"plateType\" \"%s\";\n'%os.environ['platetype'])
                f.write('fileInfo \"show\" \"%s\";\n'%os.environ['show'])
                f.write('fileInfo \"seq\" \"%s\";\n'%os.environ['seq'])
                f.write('fileInfo \"shot\" \"%s\";\n'%os.environ['shot'])
                if stereo:
                    f.write('fileInfo \"stereo\" \"true\";\n')
                else:
                    f.write('fileInfo \"stereo\" \"false\";\n')
                f.write('fileInfo \"user\" \"%s\";\n'%self.user)

                # write camera point group...

                f.write('\n')
                f.write('// create camera point group...\n')
                name = 'cam_loc'
                f.write('string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n'%name)
                f.write('$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n')
                f.write('\n')

                # write points...
                l = tde4.getPointList(self.campg)
                for p in l:
                    if tde4.isPointCalculated3D(self.campg,p):
                        name = tde4.getPointName(self.campg,p)
                        name = 'cam_%s'%self.validName(name)
                        p3d = tde4.getPointCalcPosition3D(self.campg,p)
                        p3d = self.convertZup(p3d,yup)

                        f.write('\n')
                        f.write('// create point %s...\n'%name)
                        f.write('string $locator = stringArrayToString(`spaceLocator -name %s`, \"\");\n'%name)
                        f.write('$locator = (\"|\" + $locator);\n')
                        f.write('xform -t %.15f %.15f %.15f $locator;\n'%(p3d[0],p3d[1],p3d[2]))
                        f.write('parent $locator $pointGroupName;\n')

                f.write('\n')
                f.write('xform -zeroTransformPivots -rotateOrder zxy -scale 1.000000 1.000000 1.000000 $pointGroupName;\n')

                f.write('string $camGeoGroupName = `group -em -name \"cam_geo\"`;\n')
                f.write('parent $camGeoGroupName $sceneGroupName;\n')

                chk_model_name = []

                if export_model == 1:
                    for i in tde4.get3DModelList(self.campg):
                        cnt = 0
                        model_path = tde4.get3DModelFilepath(self.campg, i)
                        model_name = '_'.join(os.path.basename(model_path).split('.')[:-1])

                        if chk_model_name.count(model_name) > 0:
                            cnt = chk_model_name.count(model_name)

                        pos = tde4.get3DModelPosition3D(self.campg, i, self.cameraList[0], 1)
                        r3d = tde4.get3DModelRotationScale3D(self.campg, i)

                        abc1 = mat3d(r3d)
                        abc2 = rot3d(mat3d(r3d)).angles(VL_APPLY_ZXY)

                        rot = self.convertToAngles(r3d)
                        #print rot

                        m_out = [[r3d[0][0], r3d[0][1], r3d[0][2], 0],
                                 [r3d[1][0], r3d[1][1], r3d[1][2], 0],
                                 [r3d[2][0], r3d[2][1], r3d[2][2], 0],
                                 [pos[0], pos[1], pos[2], 1]]

                        if os.path.isfile(model_path):
                            f.write('file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n'%(model_path))
                            f.write('$referenceOBJ = `ls -tr \"%s*\"`;\n' % model_name)
                            f.write('xform -roo zxy -matrix %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f $referenceOBJ[%s];\n' %(
                                m_out[0][0], m_out[0][1], m_out[0][2], m_out[0][3],
                                m_out[1][0], m_out[1][1], m_out[1][2], m_out[1][3],
                                m_out[2][0], m_out[2][1], m_out[2][2], m_out[2][3],
                                m_out[3][0], m_out[3][1], m_out[3][2], m_out[3][3], str(cnt)))

                            f.write('xform -ro %f %f %f $referenceOBJ[%s];\n\n' % (rot[0], rot[1], rot[2], str(cnt)))
                            if not model_name.count('spheregrid') > 0:
                                f.write('parent $referenceOBJ[%s] $camGeoGroupName;\n' % str(cnt))
                        else:
                            tde4.postQuestionRequester('Export Maya...', '%s\nfile not found. please check file path!' % model_path,'Ok')

                        chk_model_name.append(model_name)

                f.write('\n')

                # write object/mocap point groups...
                camera = tde4.getCurrentCamera()
                noframes = tde4.getCameraNoFrames(camera)
                pgl = tde4.getPGroupList()
                index = 1
                locNum = 1
                for pg in pgl:
                    if tde4.getPGroupType(pg)=='OBJECT' and camera!=None:
                        f.write('\n')
                        f.write('// create object point group...\n')
                        if tde4.get3DModelList(pg):
                            model_path = tde4.get3DModelFilepath(pg, tde4.get3DModelList(pg)[0])
                            model_name = '_'.join(os.path.basename(model_path).split('.')[:-1])
                            pgname = '%s_loc' % (model_name.split('_')[0])
                        else:
                            model_name = None
                            pgname = 'obj%d_loc' % (index)

                        index += 1
                        f.write('string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n'%pgname)
                        f.write('$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n')

                        # write points...
                        l = tde4.getPointList(pg)
                        for p in l:
                            if tde4.isPointCalculated3D(pg,p):
                                name = tde4.getPointName(pg,p)
                                name = 'obj%d_%s'%(index-1, self.validName(name))
                                p3d = tde4.getPointCalcPosition3D(pg,p)
                                p3d = self.convertZup(p3d,yup)
                                f.write('\n')
                                f.write('// create point %s...\n'%name)
                                f.write('string $locator%d = stringArrayToString(`spaceLocator -name %s`, \"\");\n'%(locNum, name))
                                f.write('$locator%d = (\"|\" + $locator%d);\n'%(locNum,locNum))
                                f.write('xform -t %.15f %.15f %.15f $locator%d;\n'%(p3d[0],p3d[1],p3d[2], locNum))
                                f.write('parent $locator%d $pointGroupName;\n'%locNum)
                                locNum += 1

                        f.write('\n')
                        scale = tde4.getPGroupScale3D(pg)
                        f.write('xform -zeroTransformPivots -rotateOrder zxy -scale %.15f %.15f %.15f $pointGroupName;\n'%(scale,scale,scale))

                        # animate object point group...
                        f.write('\n')
                        f.write('// animating point group %s...\n'%pgname)
                        frame = 1
                        model_keyframe = ''
                        while frame<=noframes:
                            # rot/pos...
                            p3d = tde4.getPGroupPosition3D(pg,camera,frame)
                            p3d = self.convertZup(p3d,yup)
                            r3d = tde4.getPGroupRotation3D(pg,camera,frame)
                            rot = self.convertToAngles(r3d)
                            if frame>1:
                                rot = [ self.angleMod360(rot0[0],rot[0]), self.angleMod360(rot0[1],rot[1]), self.angleMod360(rot0[2],rot[2]) ]
                            rot0 = rot
                            f.write('setKeyframe -at translateX -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[0]))
                            f.write('setKeyframe -at translateY -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[1]))
                            f.write('setKeyframe -at translateZ -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[2]))
                            f.write('setKeyframe -at rotateX -t %d -v %.15f $pointGroupName; '%(frame+frame0,rot[0]))
                            f.write('setKeyframe -at rotateY -t %d -v %.15f $pointGroupName; '%(frame+frame0,rot[1]))
                            f.write('setKeyframe -at rotateZ -t %d -v %.15f $pointGroupName;\n'%(frame+frame0,rot[2]))

                            frame += 1

                        if export_model == 1 and model_name:
                            f.write('string $objGeoGroupName = `group -em -name \"%s_geo\"`;\n' % (model_name.split('_')[0]))
                            for i in tde4.get3DModelList(pg):
                                model_path = tde4.get3DModelFilepath(pg, i)
                                model_name = '_'.join(os.path.basename(model_path).split('.')[:-1])

                                if os.path.isfile(model_path):
                                    f.write('file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n'%(model_path))
                                    f.write('$referenceOBJ = `ls -tr \"|%s*\"`;\n' % model_name)
                                    f.write('xform -rotateOrder zxy $referenceOBJ;\n')
                                    f.write('parentConstraint $pointGroupName $referenceOBJ;\n')
                                    f.write('parent $referenceOBJ $objGeoGroupName;\n')
                                else:
                                    tde4.postQuestionRequester('Export Maya...', '%s\nfile not found. please check file path!' % model_path,'Ok')
                            f.write('parent $objGeoGroupName $sceneGroupName;\n')

                    # mocap point groups...
                    if tde4.getPGroupType(pg)=='MOCAP' and camera!=None:
                        f.write('\n')
                        f.write('// create mocap point group...\n')
                        pgname = 'mocap%d_loc' % index
                        index += 1
                        f.write('string $pointGroupName = `group -em -name  \"%s\" -parent $sceneGroupName`;\n'%pgname)
                        f.write('$pointGroupName = ($sceneGroupName + \"|\" + $pointGroupName);\n')

                        # write points...
                        l = tde4.getPointList(pg)
                        for p in l:
                            if tde4.isPointCalculated3D(pg,p):
                                name = tde4.getPointName(pg,p)
                                name = 'mocap%d_%s'%(index-1, self.validName(name))
                                p3d = tde4.getPointMoCapCalcPosition3D(pg,p,camera,1)
                                p3d = self.convertZup(p3d,yup)
                                f.write('\n')
                                f.write('// create point %s...\n'%name)
                                f.write('string $locator = stringArrayToString(`spaceLocator -name %s`, \"\");\n'%name)
                                f.write('$locator = (\"|\" + $locator);\n')
                                f.write('xform -t %.15f %.15f %.15f $locator;\n'%(p3d[0],p3d[1],p3d[2]))
                                for frame in range(1,noframes+1):
                                    p3d = tde4.getPointMoCapCalcPosition3D(pg,p,camera,frame)
                                    p3d = self.convertZup(p3d,yup)
                                    f.write('setKeyframe -at translateX -t %d -v %.15f $locator; '%(frame+frame0,p3d[0]))
                                    f.write('setKeyframe -at translateY -t %d -v %.15f $locator; '%(frame+frame0,p3d[1]))
                                    f.write('setKeyframe -at translateZ -t %d -v %.15f $locator; '%(frame+frame0,p3d[2]))
                                f.write('parent $locator $pointGroupName;\n')

                        f.write('\n')
                        scale = tde4.getPGroupScale3D(pg)
                        f.write('xform -zeroTransformPivots -rotateOrder zxy -scale %.15f %.15f %.15f $pointGroupName;\n'%(scale,scale,scale))

                        # animate mocap point group...
                        f.write('\n')
                        f.write('// animating point group %s...\n'%pgname)
                        frame = 1
                        while frame<=noframes:
                            # rot/pos...
                            p3d = tde4.getPGroupPosition3D(pg,camera,frame)
                            p3d = self.convertZup(p3d,yup)
                            r3d = tde4.getPGroupRotation3D(pg,camera,frame)
                            rot = self.convertToAngles(r3d)
                            if frame>1:
                                rot = [ self.angleMod360(rot0[0],rot[0]), self.angleMod360(rot0[1],rot[1]), self.angleMod360(rot0[2],rot[2]) ]
                            rot0 = rot
                            f.write('setKeyframe -at translateX -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[0]))
                            f.write('setKeyframe -at translateY -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[1]))
                            f.write('setKeyframe -at translateZ -t %d -v %.15f $pointGroupName; '%(frame+frame0,p3d[2]))
                            f.write('setKeyframe -at rotateX -t %d -v %.15f $pointGroupName; '%(frame+frame0,rot[0]))
                            f.write('setKeyframe -at rotateY -t %d -v %.15f $pointGroupName; '%(frame+frame0,rot[1]))
                            f.write('setKeyframe -at rotateZ -t %d -v %.15f $pointGroupName;\n'%(frame+frame0,rot[2]))

                            frame += 1

                        if export_model == 1:
                            for i in tde4.get3DModelList(pg):
                                model_path = tde4.get3DModelFilepath(pg, i)
                                if os.path.isfile(model_path):
                                    f.write('file -reference -type \"OBJ\" -loadReferenceDepth \"all\" -mergeNamespacesOnClash true -options \"mo=1\" \"%s\";\n'%(model_path))

                # global (scene node) transformation...
                p3d = tde4.getScenePosition3D()
                p3d = self.convertZup(p3d,yup)
                r3d = tde4.getSceneRotation3D()
                rot = self.convertToAngles(r3d)
                s = tde4.getSceneScale3D()
                f.write('xform -zeroTransformPivots -rotateOrder zxy -translation %.15f %.15f %.15f -scale %.15f %.15f %.15f -rotation %.15f %.15f %.15f $sceneGroupName;\n\n'%(p3d[0],p3d[1],p3d[2],s,s,s,rot[0],rot[1],rot[2]))

                f.write('\n')
                f.close()

                path2 = tde4.getWidgetValue(self.req,'file_browser')
                if not path2.endswith('.mel'): path = path+'.mel'
                f2 = open('/tmp/tde4_exported_mel.txt', 'w')
                f2.write(path2)
                f2.close()

                tde4.postQuestionRequester('Export Maya...', 'Project successfully exported. \n OVERSCAN VALUE : ' + str(overscan), 'Ok')
            else:
                tde4.postQuestionRequester('Export Maya...', 'Error, couldn\"t open file.','Ok')
