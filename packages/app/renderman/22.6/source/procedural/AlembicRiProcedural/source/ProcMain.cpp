//-*****************************************************************************
//
// Modified for Dexter Pipe-Line
//
// LASTRELEASE
//  -2017.09.01 $1: primitive index for OpenEXRId
//                  - add object_id (auto)
//                  - add group_id by args.pid
//                support curve render
//                - args.curveTip, args.curveRoot
//
//-*****************************************************************************

#include <iostream>
#include <set>
#include <ri.h>

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreFactory/All.h>

#include "ProcArgs.h"
#include "PathUtil.h"
#include "WriteGeo.h"

#include "json.h"
#include <memory>

using namespace Alembic::AbcGeom;


class AttributeBlockHelper
{
public:
    AttributeBlockHelper( const ObjectHeader &ohead )
    {
        RiAttributeBegin();
        WriteIdentifier( ohead );
    }

    ~AttributeBlockHelper()
    {
        RiAttributeEnd();
    }
};




typedef std::auto_ptr<AttributeBlockHelper> AttributeBlockHelperAutoPtr;

//-*****************************************************************************
void WalkObject( IObject parent, const ObjectHeader &ohead, int* g_oid, ProcArgs &args,
                 PathList::const_iterator I, PathList::const_iterator E,
                 bool visible=true)
{
    AttributeBlockHelperAutoPtr blockHelper;
    if ( !args.excludeXform )
    {
        blockHelper.reset( new AttributeBlockHelper( ohead ) );
    }

    //set this if we should continue traversing
    IObject nextParentObject;

    //construct the baseObject first so that we can perform visibility
    //testing on it.
    IObject baseObject( parent, ohead.getName() );
    switch( GetVisibility(baseObject,
            ISampleSelector(args.frame / args.fps ) ) )
    {
    case kVisibilityVisible:
        visible = true;
        break;
    case kVisibilityHidden:
        visible = false;
        break;
    default:
        break;
    }

    // charles edited
    // override visibility from attr json
    json_object *jsnObj = NULL;
    if( args.atJson != NULL)
    {
        std::string jsnObj_name = ohead.getName();
        jsnObj_name = jsnObj_name.substr(jsnObj_name.find(":") + 1);

        json_object_object_get_ex(args.atJson, jsnObj_name.c_str(), &jsnObj);

        json_object *jsnAttr;
        if(jsnObj != NULL)
        {
            if(json_object_object_get_ex(jsnObj, "visibility", &jsnAttr))
                visible = json_object_get_boolean(jsnAttr);
        }


    }
    // end


    if ( IXform::matches( ohead ) )
    {
        if ( args.excludeXform )
        {
            nextParentObject = IObject( parent, ohead.getName() );
        }
        else
        {
            IXform xform( baseObject, kWrapExisting );
            ProcessXform( xform, args, jsnObj );

            nextParentObject = xform;
        }
    }
    else if ( ISubD::matches( ohead ) )
    {
        if ( !blockHelper.get() )
        {
            blockHelper.reset( new AttributeBlockHelper( ohead ) );
        }

        std::string faceSetName;

        ISubD subd( baseObject, kWrapExisting );

        //if we haven't reached the end of a specified -objectpath,
        //check to see if the next token is a faceset name.
        //If it is, send the name to ProcessSubD for addition of
        //"hole" tags for the non-matching faces
        if ( I != E )
        {
            if ( subd.getSchema().hasFaceSet( *I ) )
            {
                faceSetName = *I;
            }
        }

        if ( visible )
        {
            ProcessSubD( subd, g_oid, args, faceSetName );
            *g_oid += 1;
        }

        //if we found a matching faceset, don't traverse below
        if ( faceSetName.empty() )
        {
            nextParentObject = subd;
        }
    }
    else if ( IPolyMesh::matches( ohead ) )
    {
        if ( !blockHelper.get() )
        {
            blockHelper.reset( new AttributeBlockHelper( ohead ) );
        }

        IPolyMesh polymesh( baseObject, kWrapExisting );

        ProcessPolyMesh( polymesh, g_oid, args, jsnObj );
        *g_oid += 1;

        nextParentObject = polymesh;
    }
    else if ( INuPatch::matches( ohead ) )
    {
        if ( !blockHelper.get() )
        {
            blockHelper.reset( new AttributeBlockHelper( ohead ) );
        }

        INuPatch patch( baseObject, kWrapExisting );

        if ( visible )
        {
            ProcessNuPatch( patch, g_oid, args );
            *g_oid += 1;
        }

        nextParentObject = patch;
    }
    else if ( IPoints::matches( ohead ) )
    {
        if ( !blockHelper.get() )
        {
            blockHelper.reset( new AttributeBlockHelper( ohead ) );
        }

        IPoints points( baseObject, kWrapExisting );

        if ( visible )
        {
            //ProcessPoints( points, g_oid, args );
        }

        nextParentObject = points;
    }
    else if ( ICurves::matches( ohead ) )
    {
        if ( !blockHelper.get() )
        {
            blockHelper.reset( new AttributeBlockHelper( ohead ) );
        }

        ICurves curves( baseObject, kWrapExisting );

        if ( visible )
        {
            ProcessCurves( curves, g_oid, args );
            *g_oid += 1;
        }

        nextParentObject = curves;
    }
    else if ( IFaceSet::matches( ohead ) )
    {
        //don't complain about discovering a faceset upon traversal
    }
    else
    {
        //Don't complain but don't walk beneath other types
    }

    if ( nextParentObject.valid() )
    {
        if ( I == E )
        {
            for ( size_t i = 0; i < nextParentObject.getNumChildren() ; ++i )
            {
                if( visible == true )   // $1
                    WalkObject( nextParentObject,
                                nextParentObject.getChildHeader( i ),
                                g_oid, args, I, E, visible );
            }
        }
        else
        {
            const ObjectHeader *nextChildHeader =
                nextParentObject.getChildHeader( *I );

            if ( nextChildHeader != NULL )
            {
                if( visible == true )   // $1
                    WalkObject( nextParentObject, *nextChildHeader, g_oid, args,
                                I+1, E, visible );
            }
        }
    }

    // RiAttributeEnd will be called by blockHelper falling out of scope
    // if set.
}


#ifdef _MSC_VER
#define RIPROC_DLL_EXPORT __declspec(dllexport)
#else
#define RIPROC_DLL_EXPORT
#endif


//-*****************************************************************************
extern "C" RIPROC_DLL_EXPORT RtPointer
ConvertParameters( RtString paramstr )
{
    try
    {
	    return (RtPointer) new ProcArgs(paramstr);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Exception thrown during ProcMain ConvertParameters: ";
        std::cerr << "\"" << paramstr << "\"";
        std::cerr << " " << e.what() << std::endl;
        return 0;
    }
}

//-*****************************************************************************
extern "C" RIPROC_DLL_EXPORT RtVoid
Free( RtPointer data )
{
    delete reinterpret_cast<ProcArgs*>( data );
}

//-*****************************************************************************
extern "C" RIPROC_DLL_EXPORT RtVoid
Subdivide( RtPointer data, RtFloat detail )
{

    ProcArgs *args = reinterpret_cast<ProcArgs*>( data );

    if ( !args )
    {
        return;
    }

    if ( args->filename.empty() )
    {
        return;
    }

    try
    {
        ::Alembic::AbcCoreFactory::IFactory factory;
        IArchive archive = factory.getArchive( args->filename );

        IObject root = archive.getTop();

        PathList path;
        TokenizePath( args->objectpath, path );

        // add_charles : $1
        int g_oid = 1;

        if ( path.empty() ) //walk the entire scene
        {
            for ( size_t i = 0; i < root.getNumChildren(); ++i )
            {
                WalkObject( root, root.getChildHeader(i), &g_oid, *args,
                            path.end(), path.end() );
            }
        }
        else //walk to a location + its children
        {
            PathList::const_iterator I = path.begin();

            const ObjectHeader *nextChildHeader =
                    root.getChildHeader( *I );
            if ( nextChildHeader != NULL )
            {
                //std::cout << nextChildHeader->getName() << std::endl;
                WalkObject( root, *nextChildHeader, &g_oid, *args, I+1, path.end() );
            }
            else
            {
                std::cout << "Not found object : " << args->objectpath << std::endl;
                return;
            }
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "exception thrown during ProcMain Subdivide: "
                  << e.what() << std::endl;
    }
}
