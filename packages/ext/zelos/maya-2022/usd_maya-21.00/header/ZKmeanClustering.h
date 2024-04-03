//--------------------//
// ZKmeanClustering.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jinhyuk Bae @ Dexter Studios                  //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZKmeanClustering_h_
#define _ZKmeanClustering_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZKmeanClustering
{
	public:

		ZKmeanClustering();
		void reset();

		void setPointSet( ZPoint* posArray, int numPtc );
		void run( int numClustering, float tolerance=0.01, int maxIter=1000, const ZPointArray& initPos=ZPointArray() );

		void getClusterNum( ZIntArray& numArray );	// # of points of each cluster 
		void getClusterId(  ZIntArray& idArray );	// id of cluster of each points
		void getPointIndex( int k, ZIntArray& idArray );  // id array of each cluster 
		void getCenterPos( ZPointArray& centerPosArray ); // center pos of each cluster 
		void getCenterPos( int k, ZPoint& centerPos ); // center pos of each cluster
		void getCovMatrix( int k, ZMatrix& covMatrix ); // get covariance matrix


	private: 
	
		ZPoint* 			_points;

		ZIntArray			_clusterId;			// id for cluster  			( # of _clusterId == # of points  )
		ZIntArray			_numPtcEachCluster; // size of each cluster 	( # of _numPtcEachCluster == # of cluster )
		vector<ZIntArray>	_pointId;			// id array of each cluster ( size of vector == # of cluster )
		
		ZPointArray			_centerPos;			// center position ( # of _centerPos == # of cluster )
		ZPointArray			_centerPosNew;		// center position ( # of _centerPos == # of cluster )

		float				_tolerance;
		int					_maxIter;
		int					_numPtc;
};

ostream&
operator<<( ostream& os, const ZKmeanClustering& object );

ZELOS_NAMESPACE_END

#endif

