#ifndef _Alembic_Prman_WriteGeo_h_
#define _Alembic_Prman_WriteGeo_h_

#include <Alembic/AbcGeom/All.h>

#include "ProcArgs.h"
#include "json.h"

using namespace Alembic::AbcGeom;

void WriteIdentifier( const ObjectHeader &ohead );

void ProcessXform( IXform &xform, ProcArgs &argsf, json_object *jsnObj);

void ProcessPolyMesh(
        IPolyMesh &polymesh, int* g_oid, ProcArgs &args, json_object *jsnObj
        );
void ProcessPolyMeshVelocity(
        IPolyMeshSchema &ps, int* g_oid, int g_gid, ProcArgs &args,
        int schemeValue, std::string subdScheme, json_object *jsnObj
        );
void ProcessPolyMeshObject(
        IPolyMeshSchema &ps, int* g_oid, int g_gid, ProcArgs &args,
        int schemeValue, std::string subdScheme, json_object *jsnObj
        );

void ProcessSubD(
        ISubD &subd, int* g_oid, ProcArgs &args,
        const std::string & facesetName = "");

void ProcessNuPatch( INuPatch &patch, int* g_oid, ProcArgs &args );

void ProcessPoints( IPoints &patch, int* g_oid, ProcArgs &args );

void ProcessCurves( ICurves &curves, int* g_oid, ProcArgs &args );


#endif
