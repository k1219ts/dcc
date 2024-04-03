//----------------//
// ZMeshElement.h //
//-------------------------------------------------------//
// author: Taeyong Kim @ nVidia                          //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZMeshElement_h_
#define _ZMeshElement_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZMeshElement
{
	protected:

		ZIntArray _vert;						// vertex indices
		ZIntArray _uv;							// UV indices

	public:

		int                               id;	// group id
		ZMeshElementType::MeshElementType type; // type

	public:

		ZMeshElement();
		ZMeshElement( const ZMeshElement& source );
		ZMeshElement( ZMeshElementType::MeshElementType elementType, int numVertices );

		void reset();

		void setCount( int numVertices );

		ZMeshElement& operator=( const ZMeshElement& source );

		int& operator[]( int i ) { return _vert[i]; }
		const int& operator[]( int i ) const { return _vert[i]; }

		int& operator()( int i ) { return _uv[i]; }
		const int& operator()( int i ) const { return _uv[i]; }

		int count() const { return (int)_vert.size(); }
};

ostream&
operator<<( ostream& os, const ZMeshElement& object );

ZELOS_NAMESPACE_END

#endif

