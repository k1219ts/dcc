//-------------------//
// ZTriMeshScatter.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jaegwang Lim @ Dexter Studios                 //
// last update: 2016.06.09                               //
//-------------------------------------------------------//

#ifndef _ZTriMeshScatter_h_
#define _ZTriMeshScatter_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZTriMeshScatter
{
	private:

		// pointer to input
		ZTriMesh*     _mesh;

		// pointer to output
		ZIntArray*    _triIndices;
		ZFloat3Array* _baryCoords;

		double        _totalArea;								// surface area of the mesh

	public:

		ZIntArray triangleList;

	public:

		bool                            useOpenMP;				// use OpenMP or not
		int                             targetNumber;			// target number
		int                             randomSeed;				// random seed
		float                           diskVariance;			// Poisson disk variance
		bool                            directRadiusControl;	// direct radius control for Poisson disk sampling
		float                           dValueMin;				// density map value min for scale remapping
		float                           dValueMax;				// density map value max for scale remapping
		float                           dValueLift;				// density map value lift
		ZString                         densityMap;				// file path name of density map
		ZString                         removeMap;				// final deletion map
		float                           removeValue;			// ref. value of the removeMap
		ZSamplingMethod::SamplingMethod method;					// sampling method

	public:

		ZTriMeshScatter();
		ZTriMeshScatter( const ZTriMesh& mesh );

		void reset();

		int scatter( ZIntArray& triIndices, ZFloat3Array& baryCoords );
		int scatter( ZPointArray& samples, const int num, bool appending );		
		int scatter( ZPointArray& points, ZVectorArray& normals, const int num, bool appending );
		int scatter( ZTriMesh* mesh0, ZTriMesh* mesh1, ZPointArray& points, ZVectorArray& normals, ZVectorArray& velocities, const float dt, const int num, bool appending); 

	private:

		void _initParams();

		void _scatter01();	// zRandomBarycentric1
		void _scatter02();	// zRandomBarycentric2
		void _scatter03();	// zPoissonDiskOnUV
		void _scatter04();	// zMonteCarlo
		void _scatter05();	// zPoissonDiskOnMesh

		void _deleteByDensityMap();
		void _deleteByRemoveMap();
};

ostream&
operator<<( ostream& os, const ZTriMeshScatter& object );

ZELOS_NAMESPACE_END

#endif

