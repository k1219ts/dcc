//-------------//
// ZSkeleton.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jungmin Lee @ Dexter Studios                  //
// last update: 2017.12.13                               //
//-------------------------------------------------------//

#ifndef _ZSkeleton_h_
#define _ZSkeleton_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZSkeleton
{

	public:

        struct Joint
        {

            ZString name;
            vector<float> translation; // trans per frame
            vector<float> orientation; // rotate per frame
			vector<float> offsets = {0,0,0}; // trans offset
			vector<ZString> channels; // channel names

			vector<Joint*> child;
			Joint* parent;

			//temp
			vector<float> Xposition;
			vector<float> Yposition;
			vector<float> Zposition;
			vector<float> Xrotation;
			vector<float> Yrotation;
			vector<float> Zrotation;
	
        };

    private:

        int dispListId;
	    GLUquadricObj* upSphereObj;
	    GLUquadricObj* cylinderObj;
	    GLUquadricObj* downSphereObj;

		std::vector<ZSkeleton::Joint*> _joints;
		ZSkeleton::Joint* _root    = new ZSkeleton::Joint;
		ZSkeleton::Joint* _current = new ZSkeleton::Joint;
		ZSkeleton::Joint* _parent  = new ZSkeleton::Joint;

	private:

		enum Mode{ NONE,OFFSET,CHANNELS,JOINT,ROOT,Frames,Time,MOTIONDATA,
					KEYS,START,END }; 
		Mode _theMode;
		int _drawingMode = 2;
		float _drawingRadius = 0.4;

		int _vertIndex;
		int _channelsNum;
		int _channelTotal;
		int _channelIndex;
		ZString _tempAttr;

		vector<float> _tempTrans;
		ZVector _tempRotates;
		ZMatrix _tempMotion;

		int _boneIndex;
		int _childCount;
		int _subParentCount;
		int _lastParentCount;

		float _currentFrame;
		float _frameTime;
		float _frameUnit;
		float _frameNum;
		float _endTime;
		vector<float> _keyframes;
		int _mdataCount;



    public:

        ZSkeleton();
        virtual ~ZSkeleton();

        void reset();

        int numJoints() const;
		int numFrames() const;
		void setCurrentFrame( float frame );
		void setDrawingMode( int drawingMode );
		void setDrawingRadius( float drawingRadius);

		// return joint index in _joints
        int jointIndex( const char* jointName ) const;

		ZSkeleton::Joint* getRoot() const;

		ZString jointName( Joint* joint );
		ZString jointName( const int index );

		int childNum( const int index ) const;
		int childNum( Joint* joint ) const;

		ZSkeleton::Joint* getParent( Joint* joint ) const;

		// get child at certain index of the joint
		ZSkeleton::Joint* childAt( Joint* joint, int childIndex ) const;

		// get channels of the joint ex) Xposition, Yposition ...
		vector<ZString> getChannels( Joint* joint ) const;

		// joints from anim file do not have offset value
		// only bvh file
		vector<float> getOffsets( Joint* joint ) const;

		// whole frame
		vector<ZVector> translation( int jointIndex ) const; 
		// certain frame
		ZVector translation( int jointIndex, int frame ) const;
		ZVector translation( Joint* joint, int frame ) const;

		// whole frame
		vector<ZVector> orientation( int jointIndex ) const;
		// certain frame
		ZVector orientation( int jointIndex, int frame ) const;
		ZVector orientation( Joint* joint, int frame) const;

        void draw();
		void recursDraw( ZSkeleton::Joint* joint, int drawMode );

        bool save( const char* filePathName ) const;
        bool load( const char* filePathName );
    
	private:

        bool _save_ANIM( ofstream& fout ) const;
        bool _save_BVH( ofstream& fout ) const;
        bool _save_JSON( ofstream& fout ) const;

        bool _load_ANIM( ifstream& fin );
        bool _load_BVH( ifstream& fin );
        bool _load_JSON( ifstream& fin );

		bool _read_BVH( const char* line );
		bool _read_ANIM( const char* line);

		// store key value
		bool _process_BVH( const char* line );
		bool _process_ANIM( const char* line);

		// store joint to _joints list
		bool _readRecursive( ZSkeleton::Joint* joint );

		// rotation value to euler angle ( bvh -> anim )
		bool _eulerRecursive( Joint* joint );

		// for maya rotation
		bool _eulerFilter( Joint* joint );

		// flip angle for euler filter
		float _naive_flip_diff( float pre, float cur );

		// ( anim rotation -> bvh rotation )
		vector<float> _getAngle( float x, float y, float z ) const;

		// simplify rotation values
		float _noScience( float num ) const;
		
    private:

        void _drawCapsule( float radius, float length ) const;
		void _drawSphere ( float radius ) const;

};

ostream&
operator<<( ostream& os, const ZSkeleton& object );

ZELOS_NAMESPACE_END

#endif

