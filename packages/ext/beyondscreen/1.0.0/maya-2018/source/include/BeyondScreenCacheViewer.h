#ifndef _BeyondScreenCacheViewer_h_
#define _BeyondScreenCacheViewer_h_

#include <Common.h>

class BeyondScreenCacheViewer : public MPxLocatorNode
{
	private:

        MObject              nodeObj;
        MString              nodeName;
        MFnDependencyNode    nodeFn;
        MFnDagNode           dagNodeFn;
        MDataBlock*          blockPtr;

        bool    failed = false;
        MString cacheFile;
        int     startFrame;
        int     endFrame;
        Manager manager;

        Matrix  imgPlaneXForm;

        float   s, t;
        Vector  P, Q, R;

	public:

        static MTypeId id;
        static MString name;

        static MObject timeObj;
        static MObject cacheFileStrObj; // BeyondScreen info cache path
        static MObject imgPlaneXFormObj;
        static MObject worldViewPointObj;
        static MObject outputObj;
        static MObject frameRangeObj;
        static MObject worldScreenCenterObj;
        static MObject drawScreenMeshObj;
        static MObject screenMeshColorObj;
        static MObject drawAimingPointObj;
        static MObject aimingPointColorObj;
        static MObject drawCameraPositionObj;
        static MObject cameraPositionColorObj;

	public:

        static  void* creator();
        static  MStatus initialize();
        virtual void    postConstructor();
        virtual MStatus compute( const MPlug& plug, MDataBlock& data );

        virtual void draw( M3dView& view, const MDagPath& path,	M3dView::DisplayStyle style, M3dView::DisplayStatus displayStatus );
        virtual bool isBounded() const;
        virtual MBoundingBox boundingBox() const;
};

#endif

