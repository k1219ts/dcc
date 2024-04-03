//--------------//
// ZVoxelizer.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZVoxelizer_h_
#define _ZVoxelizer_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZVoxelizer : public ZGrid3D
{
	private:

		bool _initialState;
		bool _onCell;						// definedd at cell or node

		int   _iMax, _jMax, _kMax;			// max. possible indices
		float _h;							// cell size
		float _eps;							// epsilon of voxel space grid
		float _negRange, _posRange;			// for narrow band fast marching method

		typedef ZHeapNode<ZInt3,float> ZHEAPNODE;
		ZMinHeap<ZInt3,float> _posHeap;
		ZMaxHeap<ZInt3,float> _negHeap;

		// grid
		ZScalarField3D* _lvs;				// pointer to signed distance field
		ZVectorField3D* _vel;				// pointer to solid velocity field
		ZMarkerField3D* _stt;				// state (far, interface, updated, or trial)

		// mesh
		ZPointArray*   _vPos;				// vertex positions
		ZVectorArray*  _vVel;				// vertex velocities (displacement)
		ZInt3Array*    _v012;				// vertex connections

	public:

		ZVoxelizer();
		ZVoxelizer( const ZGrid3D& grid );
		ZVoxelizer( float h, int maxSubdivision, const ZBoundingBox& bBox );

		void reset();

		void set( const ZGrid3D& grid );
		void set( float h, int maxSubdivision, const ZBoundingBox& bBox );

		// lvs: signed distance field
		// vel: mesh displacement field
		// mesh: in world space
		// vDsp: vertex displacements in world space
		void addMesh( ZScalarField3D& lvs, ZMarkerField3D& stt, ZTriMesh& mesh, float negRange, float posRange );
		void addMesh( ZScalarField3D& lvs, ZVectorField3D& vel, ZMarkerField3D& stt, ZTriMesh& mesh, ZVectorArray& vVel, float negRange, float posRange );

		void finalize();

	private:

		void _tagInterfacialElements();
		void _update( const ZInt3& ijk, int sign );
};

ostream&
operator<<( ostream& os, const ZVoxelizer& object );

ZELOS_NAMESPACE_END

#endif

