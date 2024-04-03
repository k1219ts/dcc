#ifndef _BeyondScreenTest_h_
#define _BeyondScreenTest_h_

#include <Common.h>

class BeyondScreenTest : public MPxLocatorNode
{
	private:

        Manager manager;

        VectorArray vertexColors;

	public:

        static MTypeId id;
        static MString name;

        static MObject cameraPositionObj;
        static MObject aimPointObj;
        static MObject projectorPositionObj;
        static MObject screenMeshObj;
        static MObject screenXFormObj;
        static MObject imageFilePathNameObj;
        static MObject outputObj;

	public:

        static  void* creator();
        static  MStatus initialize();
        virtual MStatus compute( const MPlug& plug, MDataBlock& data );

        virtual void draw( M3dView& view, const MDagPath& path,	M3dView::DisplayStyle style, M3dView::DisplayStatus displayStatus );
        virtual bool isBounded() const;
};

#endif

