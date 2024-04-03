//-*****************************************************************************
//
// Modified for Dexter Pipe-Line
//
// LASTRELEASE
//  - 2017.09.01 $1: primitive index for OpenEXRId
//                  - add object_id (auto)
//                  - add group_id by args.pid
//  - 2017.09.01 $2: support curve render
//                  - args.curveTip, args.curveRoot
//  - 2017.09.08 $3: length bugfix, curve width scaling
//  - 2017.09.10 $4: group_id for crowd agent index "rman__riattr__user_Agent__Index"
//  - 2018.03.21 $5: Add json overriding for Shape
//                   Fix MatteObject
//  - 2018.05.14 $6: reverseOrientation for polygon
//
//-*****************************************************************************


#include <ri.h>

#include "WriteGeo.h"
#include "SampleUtil.h"
#include "ArbAttrUtil.h"
#include "SubDTags.h"


void ProcessXform( IXform &xform,
                   ProcArgs &args,
                   json_object *jsnObj )
{
    IXformSchema &xs = xform.getSchema();

    TimeSamplingPtr ts = xs.getTimeSampling();
    size_t xformSamps = xs.getNumSamples();

    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, xformSamps, sampleTimes );

    // edit rmantd
    ICompoundProperty arbGeomAttributes = xs.getArbGeomParams();
    SampleTimeSet::iterator first = sampleTimes.begin();
    ISampleSelector firstSelector( *first );

    bool multiSample = sampleTimes.size() > 1;

    std::vector<XformSample> sampleVectors;
    sampleVectors.resize( sampleTimes.size() );

    //fetch all operators at each sample time first
    size_t sampleTimeIndex = 0;
    for ( SampleTimeSet::iterator I = sampleTimes.begin();
          I != sampleTimes.end(); ++I, ++sampleTimeIndex )
    {
        ISampleSelector sampleSelector( *I );

        xs.get( sampleVectors[sampleTimeIndex], sampleSelector );
    }


    if (xs.getInheritsXforms () == false)
    {
        RiIdentity ();
    }

    // edit rmantd : STR to matrix
    std::vector<XformSample> concat;
    concat.resize( sampleTimes.size() );
    for ( size_t i = 0, e = xs.getNumOps(); i < e; ++i )
    {
        for ( size_t j = 0; j < sampleVectors.size(); ++j )
        {
            XformOp &op = sampleVectors[j][i];
            switch ( op.getType() )
            {
            case kScaleOperation:
            {
                V3d value = op.getScale();
                concat[j].setScale( value );
                break;
            }
            case kTranslateOperation:
            {
                V3d value = op.getTranslate();
                concat[j].setTranslation( value );
                break;
            }
            case kRotateOperation:
            case kRotateXOperation:
            case kRotateYOperation:
            case kRotateZOperation:
            {
                V3d axis = op.getAxis();
                float degrees = op.getAngle();
                if( axis.x == 1 ) concat[j].setXRotation( degrees );
                if( axis.y == 1 ) concat[j].setYRotation( degrees );
                if( axis.z == 1 ) concat[j].setZRotation( degrees );
                break;
            }
            case kMatrixOperation:
            {
                concat[j].setMatrix( op.getMatrix() );
                break;
            }
            }
        }
    }

    // Ri
    if( multiSample ) { WriteMotionBegin( args, sampleTimes ); }

    for( size_t j=0; j<sampleVectors.size(); ++j )
    {
        WriteConcatTransform( concat[j].getMatrix() );
    }

    if( multiSample ) { RiMotionEnd(); }

    // edit rmantd, charles edited $5
    if(jsnObj != NULL)
        AddArbitraryGeomAttributes_fromJson( arbGeomAttributes, args, jsnObj);
    else
        AddArbitraryGeomAttributes( xform, arbGeomAttributes, firstSelector, args);
}

//-*****************************************************************************
void ProcessPolyMesh( IPolyMesh &polymesh,
                      int* g_oid, ProcArgs &args,
                      json_object *jsnObj )
{
    IPolyMeshSchema &ps = polymesh.getSchema();

    TimeSamplingPtr ts = ps.getTimeSampling();

    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, ps.getNumSamples(), sampleTimes );

    //--------------------------------------------------------------------------
    // Attribute Block
    SampleTimeSet::iterator first = sampleTimes.begin();
    ISampleSelector firstSelector( *first );
    //
    IPolyMeshSchema::Sample sample = ps.getValue( firstSelector );
    Abc::V3fArraySamplePtr velValuePtr = sample.getVelocities();

    ICompoundProperty geomAttributes = ps.getArbGeomParams();

    // get attributes for subdiv
    int schemeValue = 100;
    std::string subdScheme;

    // set subdivScheme
    if(args.subdiv == 1)
    {
        const char *attrName = "rman__torattr___subdivScheme";
        // charles edited $5
        if( jsnObj != NULL )
        {
            json_object *jsnAttr;
            if( json_object_object_get_ex(jsnObj, attrName, &jsnAttr) )
                schemeValue = json_object_get_int(jsnAttr);
        }
        else
        {
            // get attributes original
            GetIntAttributeValue(geomAttributes, firstSelector, attrName, &schemeValue);
        }
    }
    // end

    if(schemeValue == 0) subdScheme = std::string("catmull-clark");
    else subdScheme = std::string("loop");

    // set gid
    // add_rmantd $4
    int agent_id = 0;
    if(args.pid)
    {
        const char *attrName = "rman__riattr__user_Agent__Index";
        // charles edited $5
        if( jsnObj != NULL )
        {
            json_object *jsnAttr;
            if( json_object_object_get_ex(jsnObj, attrName, &jsnAttr) )
                agent_id = json_object_get_int(jsnAttr);
        }
        else
        {
            // get attributes original
            GetIntAttributeValue(geomAttributes, firstSelector, attrName, &agent_id);
        }
    }

    int g_gid = args.pid + agent_id;

    // add randerman attributes
    // charles edited $5
    if( jsnObj != NULL)
        AddRmanAttributes_fromJson( args, jsnObj );
    else
        AddRmanAttributes(geomAttributes, firstSelector);

    AddRlfInjectStructure(geomAttributes, firstSelector);
    //--------------------------------------------------------------------------

    if( velValuePtr && args.dt > 0.0 ) {
        ProcessPolyMeshVelocity( ps, g_oid, g_gid, args,
                                 schemeValue, subdScheme, jsnObj );
    } else {
        ProcessPolyMeshObject( ps, g_oid, g_gid, args,
                               schemeValue, subdScheme, jsnObj );
    }
}

void ProcessPolyMeshVelocity( IPolyMeshSchema &ps,
                              int* g_oid, int g_gid,
                              ProcArgs &args,
                              int schemeValue, std::string subdScheme,
                              json_object *jsnObj )
{
    TimeSamplingPtr ts = ps.getTimeSampling();
    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, ps.getNumSamples(), sampleTimes );

    bool multiSample = sampleTimes.size() > 1;

    SampleTimeSet::iterator iter = sampleTimes.begin();
    ISampleSelector sampleSelector( *iter );
    IPolyMeshSchema::Sample sample = ps.getValue( sampleSelector );
    RtInt npolys = (RtInt) sample.getFaceCounts()->size();

    ParamListBuilder paramListBuilder;
    paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );

    // velocity
    std::vector<float> nextPos;
    nextPos.resize( sample.getPositions()->size()*3 );
    Abc::P3fArraySamplePtr posValuePtr = sample.getPositions();
    Abc::V3fArraySamplePtr velValuePtr = sample.getVelocities();

    if( velValuePtr->valid() )
    {
        const int sizeVel = velValuePtr->size();
        const int sizePos = posValuePtr->size();
        if( sizeVel != sizePos ) {
            std::cout << "different size between position & velocity" << std::endl;
            int idx = 0;
            for( int v=0; v<sizePos; ++v )
            {
                const Abc::V3f& pos = posValuePtr->get()[v];
                nextPos[idx=(3*v)] = pos.x;
                nextPos[  ++idx  ] = pos.y;
                nextPos[  ++idx  ] = pos.z;
            }
        } else {
            int idx = 0;
            for( int v=0; v<sizePos; ++v )
            {
                const Abc::V3f& pos = posValuePtr->get()[v];
                const Abc::V3f& vel = velValuePtr->get()[v];
                nextPos[idx=(3*v)] = pos.x + args.dt * vel.x;
                nextPos[  ++idx  ] = pos.y + args.dt * vel.y;
                nextPos[  ++idx  ] = pos.z + args.dt * vel.z;
            }
        }
    } else {
        std::cout << "Invalid to velocity" << std::endl;
    }

    IV2fGeomParam uvParam = ps.getUVsParam();
    if( uvParam.valid() )
    {
        ICompoundProperty parent = uvParam.getParent();
        if( !args.flipv )
        {
            AddGeomParamToParamListBuilder<IV2fGeomParam>(
                    parent, uvParam.getHeader(), sampleSelector,
                    "float", paramListBuilder, 2, "st" );
        }
        else if( std::vector<float> * values =
                AddGeomParamToParamListBuilderAsFloat<IV2fGeomParam, float>(
                    parent, uvParam.getHeader(), sampleSelector,
                    "float", paramListBuilder, "st" ) )
        {
            for( size_t i=1, e=values->size(); i<e; i+=2 ) {
                (*values)[i] = 1.0 - (*values)[i];
            }
        }
    }

    ICompoundProperty arbGeomParams = ps.getArbGeomParams();

    if( jsnObj != NULL )
    {
        AddArbitraryGeomParams_fromJson( arbGeomParams, sampleSelector,
                                         jsnObj, paramListBuilder );
    }
    else
        AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

    // add_rmantd : $1
    if(args.pid && *g_oid > 0)
    {
        paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
        paramListBuilder.addAsInt("constant int group_id", &g_gid, 1);
        // paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
    }


    if( multiSample ) RiMotionBegin( 2, args.shutterOpen, args.shutterClose );

    if( schemeValue == 100 )
    {
        IN3fGeomParam nParam = ps.getNormalsParam();
        if( nParam.valid() )
        {
            ICompoundProperty parent = nParam.getParent();
            AddGeomParamToParamListBuilder<IN3fGeomParam>(
                    parent,
                    nParam.getHeader(),
                    sampleSelector,
                    "normal",
                    paramListBuilder );
        }

        std::string typeValue = std::string("mesh");
        const char * stValue[] = {typeValue.c_str()};
        paramListBuilder.add( "constant string primtype", (RtPointer)stValue );

        RiPointsPolygonsV(
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );

        if( multiSample )
        {
            paramListBuilder.add( "P", &nextPos[0] );

            RiPointsPolygonsV(
                    npolys,
                    (RtInt*) sample.getFaceCounts()->get(),
                    (RtInt*) sample.getFaceIndices()->get(),
                    paramListBuilder.n(),
                    paramListBuilder.nms(),
                    paramListBuilder.vals() );
        }
    }
    else
    {
        SubDTagBuilder tags;
        tags.add( "facevaryinginterpolateboundary" );
        tags.addIntArg( 1 );
        tags.add( "interpolateboundary" );
        tags.addIntArg( 1 );
        tags.add( "facevaryingpropagatecorners" );
        tags.addIntArg( 0 );
        std::string typeValue = std::string("subdiv");
        const char * stValue[] = {typeValue.c_str()};
        paramListBuilder.add( "constant string primtype", (RtPointer)stValue );

        RiHierarchicalSubdivisionMeshV(
            const_cast<RtToken>( subdScheme.c_str() ),
            npolys,
            (RtInt*) sample.getFaceCounts()->get(),
            (RtInt*) sample.getFaceIndices()->get(),
            tags.nt(), tags.tags(), tags.nargs( true ),
            tags.intargs(), tags.floatargs(), tags.stringargs(),
            paramListBuilder.n(),
            paramListBuilder.nms(),
            paramListBuilder.vals() );

        if( multiSample )
        {
            paramListBuilder.add( "P", &nextPos[0] );

            RiHierarchicalSubdivisionMeshV(
                const_cast<RtToken>( subdScheme.c_str() ),
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                tags.nt(), tags.tags(), tags.nargs( true ),
                tags.intargs(), tags.floatargs(), tags.stringargs(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );
        }
    }
    if( multiSample ) RiMotionEnd();
}

void ProcessPolyMeshObject( IPolyMeshSchema &ps,
                            int* g_oid, int g_gid,
                            ProcArgs &args,
                            int schemeValue, std::string subdScheme,
                            json_object *jsnObj )
{
    std::vector<IFaceSet> faceSets;
    std::vector<std::string> faceSetResourceNames;

    TimeSamplingPtr ts = ps.getTimeSampling();
    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, ps.getNumSamples(), sampleTimes );

    bool multiSample = sampleTimes.size() > 1;


    if ( multiSample ) { WriteMotionBegin( args, sampleTimes ); }

    for ( SampleTimeSet::iterator iter = sampleTimes.begin();
          iter != sampleTimes.end(); ++ iter )
    {
        ISampleSelector sampleSelector( *iter );

        IPolyMeshSchema::Sample sample = ps.getValue( sampleSelector );
        RtInt npolys = (RtInt) sample.getFaceCounts()->size();

        ParamListBuilder paramListBuilder;

        paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );

        IV2fGeomParam uvParam = ps.getUVsParam();
        if ( uvParam.valid() )
        {
            ICompoundProperty parent = uvParam.getParent();
            if ( !args.flipv )
            {
                AddGeomParamToParamListBuilder<IV2fGeomParam>(
                    parent, uvParam.getHeader(), sampleSelector,
                    "float", paramListBuilder, 2, "st");
            }
            else if ( std::vector<float> * values =
                    AddGeomParamToParamListBuilderAsFloat<IV2fGeomParam, float>(
                        parent, uvParam.getHeader(), sampleSelector,
                        "float", paramListBuilder, "st") )
            {
                for ( size_t i = 1, e = values->size(); i < e; i += 2 ) {
                    (*values)[i] = 1.0 - (*values)[i];
                }
            }
        }

        ICompoundProperty arbGeomParams = ps.getArbGeomParams();

        if( jsnObj != NULL )
        {
            AddArbitraryGeomParams_fromJson( arbGeomParams, sampleSelector,
                                             jsnObj, paramListBuilder );
        }
        else
            AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

        // add_rmantd : $1
        if(args.pid && *g_oid > 0)
        {
            paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
            paramListBuilder.addAsInt("constant int group_id", &g_gid, 1);
            // paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
        }

        if( schemeValue == 100 )
        {
            IN3fGeomParam nParam = ps.getNormalsParam();
            if ( nParam.valid() )
            {
                ICompoundProperty parent = nParam.getParent();
                AddGeomParamToParamListBuilder<IN3fGeomParam>(
                    parent,
                    nParam.getHeader(),
                    sampleSelector,
                    "normal",
                    paramListBuilder);

            }

            std::string typeValue = std::string("mesh");
            const char * stValue[] = {typeValue.c_str()};
            paramListBuilder.add( "constant string primtype", (RtPointer)stValue );

            RiPointsPolygonsV(
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );
        }
        else
        {
            // $4
            SubDTagBuilder tags;
            tags.add( "facevaryinginterpolateboundary" );
            tags.addIntArg( 1 );
            tags.add( "interpolateboundary" );
            tags.addIntArg( 1 );
            tags.add( "facevaryingpropagatecorners" );
            tags.addIntArg( 0 );

            std::string typeValue = std::string("subdiv");
            const char * stValue[] = {typeValue.c_str()};
            paramListBuilder.add( "constant string primtype", (RtPointer)stValue );

            RiHierarchicalSubdivisionMeshV(
                const_cast<RtToken>( subdScheme.c_str() ),
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                tags.nt(),
                tags.tags(),
                tags.nargs( true ),
                tags.intargs(),
                tags.floatargs(),
                tags.stringargs(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );
        }
    }
    if (multiSample) RiMotionEnd();
}

//-*****************************************************************************
void ProcessSubD( ISubD &subd, int* g_oid, ProcArgs &args, const std::string & facesetName )
{
    ISubDSchema &ss = subd.getSchema();

    TimeSamplingPtr ts = ss.getTimeSampling();

    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, ss.getNumSamples(), sampleTimes );

    bool multiSample = sampleTimes.size() > 1;

    //include this code path for future expansion
    bool isHierarchicalSubD = false;
    bool hasLocalResources = false;

    std::vector<IFaceSet> faceSets;
    std::vector<std::string> faceSetResourceNames;
    if ( facesetName.empty() )
    {
        std::vector <std::string> childFaceSetNames;
        ss.getFaceSetNames(childFaceSetNames);

        faceSets.reserve(childFaceSetNames.size());
        faceSetResourceNames.reserve(childFaceSetNames.size());

        for (size_t i = 0; i < childFaceSetNames.size(); ++i)
        {
            faceSets.push_back(ss.getFaceSet(childFaceSetNames[i]));

            IFaceSet & faceSet = faceSets.back();

            std::string resourceName = args.getResource(
                    faceSet.getFullName() );

            if ( resourceName.empty() )
            {
                resourceName = args.getResource( faceSet.getName() );
            }

            faceSetResourceNames.push_back(resourceName);

        }
    }

    if ( multiSample ) { WriteMotionBegin( args, sampleTimes ); }

    for ( SampleTimeSet::iterator iter = sampleTimes.begin();
          iter != sampleTimes.end(); ++iter )
    {

        ISampleSelector sampleSelector( *iter );

        ISubDSchema::Sample sample = ss.getValue( sampleSelector );

        RtInt npolys = (RtInt) sample.getFaceCounts()->size();

        ParamListBuilder paramListBuilder;

        paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );

        IV2fGeomParam uvParam = ss.getUVsParam();
        if ( uvParam.valid() )
        {
            ICompoundProperty parent = uvParam.getParent();

            if ( !args.flipv )
            {
                AddGeomParamToParamListBuilder<IV2fGeomParam>(
                    parent, uvParam.getHeader(), sampleSelector,
                    "float", paramListBuilder, 2, "st");
            }
            else if ( std::vector<float> * values =
                    AddGeomParamToParamListBuilderAsFloat<IV2fGeomParam, float>(
                        parent, uvParam.getHeader(), sampleSelector,
                        "float", paramListBuilder, "st") )
            {
                for ( size_t i = 1, e = values->size(); i < e; i += 2 )
                {
                    (*values)[i] = 1.0 - (*values)[i];
                }
            }

        }

        ICompoundProperty arbGeomParams = ss.getArbGeomParams();

        AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

        // add_rmantd : $1
        if(args.pid && *g_oid > 0)
        {
            paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
            paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
        }

        std::string subdScheme = sample.getSubdivisionScheme();

        SubDTagBuilder tags;

        ProcessFacevaryingInterpolateBoundry( tags, sample );
        ProcessInterpolateBoundry( tags, sample );
        ProcessFacevaryingPropagateCorners( tags, sample );
        ProcessHoles( tags, sample );
        ProcessCreases( tags, sample );
        ProcessCorners( tags, sample );

        if ( !facesetName.empty() )
        {
            if ( ss.hasFaceSet( facesetName ) )
            {
                IFaceSet faceSet = ss.getFaceSet( facesetName );

                // TODO, move the hold test outside of MotionBegin
                // as it's not meaningful to change per sample

                IFaceSetSchema::Sample faceSetSample =
                        faceSet.getSchema().getValue( sampleSelector );

                std::set<int> facesToKeep;

                facesToKeep.insert( faceSetSample.getFaces()->get(),
                        faceSetSample.getFaces()->get() +
                                faceSetSample.getFaces()->size() );

                for ( int i = 0; i < npolys; ++i )
                {
                    if ( facesToKeep.find( i ) == facesToKeep.end() )
                    {
                        tags.add( "hole" );
                        tags.addIntArg( i );
                    }
                }
            }
        }
        else
        {
            //loop through the facesets and determine whether there are any
            //resources assigned to each

            for (size_t i = 0; i < faceSetResourceNames.size(); ++i)
            {
                const std::string & resourceName = faceSetResourceNames[i];

                //TODO, visibility?

                if ( !resourceName.empty() )
                {
                    IFaceSet & faceSet = faceSets[i];

                    isHierarchicalSubD = true;

                    tags.add("faceedit");

                    Int32ArraySamplePtr faces = faceSet.getSchema().getValue(
                            sampleSelector ).getFaces();

                    for (size_t j = 0, e = faces->size(); j < e; ++j)
                    {
                        tags.addIntArg(1); //yep, every face gets a 1 in front of it too
                        tags.addIntArg( (int) faces->get()[j]);
                    }

                    tags.addStringArg( "attributes" );
                    tags.addStringArg( resourceName );
                    tags.addStringArg( "shading" );
                }
            }
        }


        if ( isHierarchicalSubD )
        {
            RiHierarchicalSubdivisionMeshV(
                const_cast<RtToken>( subdScheme.c_str() ),
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                tags.nt(),
                tags.tags(),
                tags.nargs( true ),
                tags.intargs(),
                tags.floatargs(),
                tags.stringargs(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals()
                                          );
        }
        else
        {
            RiSubdivisionMeshV(
                const_cast<RtToken>(subdScheme.c_str() ),
                npolys,
                (RtInt*) sample.getFaceCounts()->get(),
                (RtInt*) sample.getFaceIndices()->get(),
                tags.nt(),
                tags.tags(),
                tags.nargs( false ),
                tags.intargs(),
                tags.floatargs(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals()
                              );
        }
    }

    if ( multiSample ) { RiMotionEnd(); }

    if ( hasLocalResources ) { RiResourceEnd(); }
}

//-*****************************************************************************
void ProcessNuPatch( INuPatch &patch, int* g_oid, ProcArgs &args )
{
    INuPatchSchema &ps = patch.getSchema();

    TimeSamplingPtr ts = ps.getTimeSampling();

    SampleTimeSet sampleTimes;
    GetRelevantSampleTimes( args, ts, ps.getNumSamples(), sampleTimes );


    //trim curves are described outside the motion blocks
    if ( ps.hasTrimCurve() )
    {
        //get the current time sample independent of any shutter values
        INuPatchSchema::Sample sample = ps.getValue(
                ISampleSelector( args.frame / args.fps ) );

        RiTrimCurve( sample.getTrimNumCurves()->size(), //numloops
                (RtInt*) sample.getTrimNumCurves()->get(),
                (RtInt*) sample.getTrimOrders()->get(),
                (RtFloat*) sample.getTrimKnots()->get(),
                (RtFloat*) sample.getTrimMins()->get(),
                (RtFloat*) sample.getTrimMaxes()->get(),
                (RtInt*) sample.getTrimNumVertices()->get(),
                (RtFloat*) sample.getTrimU()->get(),
                (RtFloat*) sample.getTrimV()->get(),
                (RtFloat*) sample.getTrimW()->get() );
    }

    bool multiSample = sampleTimes.size() > 1;

    if ( multiSample ) { WriteMotionBegin( args, sampleTimes ); }

    for ( SampleTimeSet::iterator iter = sampleTimes.begin();
          iter != sampleTimes.end(); ++iter )
    {
        ISampleSelector sampleSelector( *iter );

        INuPatchSchema::Sample sample = ps.getValue( sampleSelector );


        ParamListBuilder paramListBuilder;

        //build this here so that it's still in scope when RiNuPatchV is
        //called.
        std::vector<RtFloat> pwValues;

        if ( sample.getPositionWeights() )
        {
            if ( sample.getPositionWeights()->size() == sample.getPositions()->size() )
            {
                //need to combine P with weight form Pw
                pwValues.reserve( sample.getPositions()->size() * 4 );

                const float32_t * pStart = reinterpret_cast<const float32_t * >(
                        sample.getPositions()->get() );
                const float32_t * wStart = reinterpret_cast<const float32_t * >(
                        sample.getPositionWeights()->get() );

                for ( size_t i = 0, e = sample.getPositionWeights()->size();
                        i < e;  ++i )
                {
                    pwValues.push_back( pStart[i*3] );
                    pwValues.push_back( pStart[i*3+1] );
                    pwValues.push_back( pStart[i*3+2] );
                    pwValues.push_back( wStart[i] );
                }

                paramListBuilder.add( "Pw", (RtPointer) &pwValues[0] );
            }
        }

        if ( pwValues.empty() )
        {
            //no Pw so go straight with P
            paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );
        }

        ICompoundProperty arbGeomParams = ps.getArbGeomParams();
        AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

        // add_rmantd : $1
        if(args.pid && *g_oid > 0)
        {
            paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
            paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
        }

        //For now, use the last knot value for umin and umax as it's
        //not described in the alembic data

        RiNuPatchV(
                sample.getNumU(),
                sample.getUOrder(),
                (RtFloat *) sample.getUKnot()->get(),
                0.0, //umin
                sample.getUKnot()->get()[sample.getUKnot()->size()-1],//umax
                sample.getNumV(),
                sample.getVOrder(),
                (RtFloat *) sample.getVKnot()->get(),
                0.0, //vmin
                sample.getVKnot()->get()[sample.getVKnot()->size()-1], //vmax
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );
    }

    if ( multiSample ) { RiMotionEnd(); }


}

//-*****************************************************************************
void ProcessPoints( IPoints &points, int* g_oid, ProcArgs &args )
{
    IPointsSchema &ps = points.getSchema();
    TimeSamplingPtr ts = ps.getTimeSampling();

    SampleTimeSet sampleTimes;

    if ( ps.getIdsProperty().isConstant() )
    {
        //grab only the current time
        sampleTimes.insert( args.frame / args.fps );
    }
    else
    {
         GetRelevantSampleTimes( args, ts, ps.getNumSamples(), sampleTimes );
    }

    bool multiSample = sampleTimes.size() > 1;

    if ( multiSample ) { WriteMotionBegin( args, sampleTimes ); }

    for ( SampleTimeSet::iterator iter = sampleTimes.begin();
          iter != sampleTimes.end(); ++iter )
    {
        ISampleSelector sampleSelector( *iter );

        IPointsSchema::Sample sample = ps.getValue( sampleSelector );


        ParamListBuilder paramListBuilder;
        paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );

        ICompoundProperty arbGeomParams = ps.getArbGeomParams();
        AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

        // add_rmantd : $1
        if(args.pid && *g_oid > 0)
        {
            paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
            paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
        }

        RiPointsV(sample.getPositions()->size(),
                paramListBuilder.n(),
                paramListBuilder.nms(),
                paramListBuilder.vals() );
    }

    if ( multiSample ) { RiMotionEnd(); }

}

//-*****************************************************************************
void ProcessCurves( ICurves &curves, int* g_oid, ProcArgs &args )
{
    ICurvesSchema &cs = curves.getSchema();
    TimeSamplingPtr ts = cs.getTimeSampling();

    SampleTimeSet sampleTimes;

    GetRelevantSampleTimes( args, ts, cs.getNumSamples(), sampleTimes );

    bool multiSample = sampleTimes.size() > 1;

    bool firstSample = true;

    // add_rmantd $3
    SampleTimeSet::iterator first = sampleTimes.begin();
    ISampleSelector firstSelector(*first);
    float rootWidth = 0.1;
    float tipWidth = 0.1;
    getCurveWidthAttributes(cs, firstSelector, &rootWidth, &tipWidth);
    // std::cout << "rootWidth : " << rootWidth << " tipWidth : " << tipWidth << std::endl;
    rootWidth *= args.curveRoot;
    tipWidth *= args.curveTip;

    for ( SampleTimeSet::iterator iter = sampleTimes.begin();
          iter != sampleTimes.end(); ++iter )
    {
        ISampleSelector sampleSelector( *iter );

        ICurvesSchema::Sample sample = cs.getValue( sampleSelector );

        //need to set the basis prior to the MotionBegin block
        if ( firstSample )
        {
            firstSample = false;

            BasisType basisType = sample.getBasis();
            if ( basisType != kNoBasis )
            {
                RtBasis * basis = NULL;
                RtInt step = 0;

                switch ( basisType )
                {
                case kBezierBasis:
                    basis = &RiBezierBasis;
                    step = RI_BEZIERSTEP;
                    break;
                case kBsplineBasis:
                    basis = &RiBSplineBasis;
                    step = RI_BSPLINESTEP;
                    break;
                case kCatmullromBasis:
                    basis = &RiCatmullRomBasis;
                    step = RI_CATMULLROMSTEP;
                    break;
                case kHermiteBasis:
                    basis = &RiHermiteBasis;
                    step = RI_HERMITESTEP;
                    break;
                case kPowerBasis:
                    basis = &RiPowerBasis;
                    step = RI_POWERSTEP;
                    break;
                default:
                    break;
                }

                if ( basis != NULL )
                {
                    RiBasis( *basis, step, *basis, step);
                }
            }


            if ( multiSample ) { WriteMotionBegin( args, sampleTimes ); }
        }

        ParamListBuilder paramListBuilder;

        // paramListBuilder.add( "P", (RtPointer)sample.getPositions()->get() );
        // IFloatGeomParam widthParam = cs.getWidthsParam();

        // add_rmantd : $3
        Abc::P3fArraySamplePtr posValuePtr = sample.getPositions();
        const int pos_size = posValuePtr->size();

        std::vector<float> points;

        const Abc::V3f& f_pos = posValuePtr->get()[0];
        // first add 1
        points.push_back(static_cast<float>(f_pos.x));
        points.push_back(static_cast<float>(f_pos.y));
        points.push_back(static_cast<float>(f_pos.z));
        // first add 2
        points.push_back(static_cast<float>(f_pos.x));
        points.push_back(static_cast<float>(f_pos.y));
        points.push_back(static_cast<float>(f_pos.z));

        for(int i=0; i < pos_size; ++i)
        {
            const Abc::V3f& pos = posValuePtr->get()[i];
            points.push_back(static_cast<float>(pos.x));
            points.push_back(static_cast<float>(pos.y));
            points.push_back(static_cast<float>(pos.z));
        }

        const Abc::V3f& e_pos = posValuePtr->get()[pos_size-1];
        // end add 1
        points.push_back(static_cast<float>(e_pos.x));
        points.push_back(static_cast<float>(e_pos.y));
        points.push_back(static_cast<float>(e_pos.z));
        // end add 2
        points.push_back(static_cast<float>(e_pos.x));
        points.push_back(static_cast<float>(e_pos.y));
        points.push_back(static_cast<float>(e_pos.z));

        paramListBuilder.add("P", &points[0]);

        int numvertices = sample.getCurvesNumVertices()->get()[0];

        std::vector<float> width;
        std::string widthName;
        int widthNum;

        if(rootWidth == tipWidth)
        {
            widthName = "constantwidth";
            widthNum = 1;
            width.push_back(static_cast<float>(rootWidth));
        }
        else
        {
            widthName = "varying float width";
            widthNum = numvertices + 2;
            width.push_back(static_cast<float>(rootWidth));
            float increase = (float)(tipWidth - rootWidth) / (float)(numvertices-1);
            for(int i=0; i < numvertices; i++)
            {
                width.push_back(static_cast<float>(rootWidth + (increase *(float)i)));
            }
            width.push_back(static_cast<float>(tipWidth));
        }

        paramListBuilder.addAsFloat(widthName, &width[0], widthNum);

        // IN3fGeomParam nParam = cs.getNormalsParam();
        // if ( nParam.valid() )
        // {
        //     ICompoundProperty parent = nParam.getParent();
        //
        //     AddGeomParamToParamListBuilder<IN3fGeomParam>(
        //             parent, nParam.getHeader(), sampleSelector,
        //             "normal", paramListBuilder);
        // }

        // IV2fGeomParam uvParam = cs.getUVsParam();
        // if ( uvParam.valid() )
        // {
        //     ICompoundProperty parent = uvParam.getParent();
        //
        //     AddGeomParamToParamListBuilder<IV2fGeomParam>(
        //             parent, uvParam.getHeader(), sampleSelector,
        //             "float", paramListBuilder, 2, "st");
        // }

        // ICompoundProperty arbGeomParams = cs.getArbGeomParams();
        // AddArbitraryGeomParams( arbGeomParams, sampleSelector, paramListBuilder );

        // add_rmantd : $1
        if(args.pid && *g_oid > 0)
        {
            paramListBuilder.addAsInt("constant int object_id", g_oid, 1);
            paramListBuilder.addAsInt("constant int group_id", &args.pid, 1);
        }
        paramListBuilder.addAsInt("constant int index", g_oid, 1);

        RtToken curveType;
        switch ( sample.getType() )
        {
            case kCubic:
                curveType = const_cast<RtToken>( "cubic" );
                break;
            default:
                curveType = const_cast<RtToken>( "linear" );
        }

        RtToken wrap;
        switch ( sample.getWrap() )
        {
            case kPeriodic:
                wrap = const_cast<RtToken>( "periodic" );
                break;
            default:
                wrap = const_cast<RtToken>( "nonperiodic" );
        }

        int num = points.size() / 3;
        RiCurvesV(curveType, sample.getNumCurves(),
                  &num,
                  wrap,
                  paramListBuilder.n(),
                  paramListBuilder.nms(),
                  paramListBuilder.vals());

    }

    if ( multiSample ) { RiMotionEnd(); }

}



//-*****************************************************************************
void WriteIdentifier( const ObjectHeader &ohead )
{
    std::string name = ohead.getFullName();

    RtToken token = RtUString(name.c_str()).CStr();
    // RiAttribute(RI_IDENTIFIER, RI_NAME, &token, RI_NULL);
    RiAttribute(const_cast<char*>("identifier"), const_cast<char*>("name"), &token, RI_NULL);
}
