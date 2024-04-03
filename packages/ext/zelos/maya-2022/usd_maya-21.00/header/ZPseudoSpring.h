//-----------------//
// ZPseudoSpring.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.28                               //
//-------------------------------------------------------//

#ifndef _ZPseudoSpring_h_
#define _ZPseudoSpring_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZPseudoSpring
{
	public:

		float   k;		///< The stiffness coefficient.
		float   d;		///< The damping coefficient.
		float   a;		///< The limit of displacement from the goal position to the current position.
		float   dt;		///< The time step size.
		ZVector f;		///< The external force.

	protected:

		ZPoint  _p0;	///< The previous position.
		ZPoint  _p;		///< The current position.
		ZVector _v;		///< The current velocity.

	public:

		ZPseudoSpring();
		ZPseudoSpring( const ZPseudoSpring& source );
		ZPseudoSpring( const ZPoint& initialPosition );

		ZPseudoSpring& operator=( const ZPseudoSpring& source );

		ZPoint position() const;
		ZVector velocity() const;

		void reInitialize( const ZPoint& goalPosition );
		ZPoint update( const ZPoint& goalPosition );
};

ostream& operator<<( ostream& os, const ZPseudoSpring& object );

ZELOS_NAMESPACE_END

#endif

