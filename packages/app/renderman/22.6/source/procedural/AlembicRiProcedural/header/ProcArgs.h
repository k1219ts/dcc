//-*****************************************************************************
//
//-*****************************************************************************

#ifndef _Alembic_Prman_ProcArgs_h_
#define _Alembic_Prman_ProcArgs_h_

#define PRMAN_USE_ABCMATERIAL


#include <string>
#include <map>
#include <vector>

#include <ri.h>

#include <boost/shared_ptr.hpp>

#include "json.h"


//-*****************************************************************************
struct ProcArgs
{
    //constructor parses
    ProcArgs( RtString paramStr, bool fromReference = false );

    //copy constructor
    ProcArgs( const ProcArgs &rhs )
    : filename( rhs.filename )
    , objectpath( rhs.objectpath )
    , frame( rhs.frame )
    , fps( rhs.fps )
    , shutterOpen( rhs.shutterOpen )
    , shutterClose( rhs.shutterClose )
    , dt( rhs.dt )
    , excludeXform( false )
    , flipv( false )
    , subdiv( rhs.subdiv )
    , subframe( rhs.subframe )
    , oid( rhs.oid )
    , gid( rhs.gid )
    , pid( rhs.pid )
    , txvar( rhs.txvar )
    , primvar( rhs.primvar )
    , cycle( rhs.cycle )
    , atJson( rhs.atJson )
    , curveTip( rhs.curveTip )
    , curveRoot( rhs.curveRoot )
    
    , filename_defined(false)
    , objectpath_defined(false)
    , frame_defined(false)
    , fps_defined(false)
    , shutterOpen_defined(false)
    , shutterClose_defined(false)
    , dt_defined(false)
    , excludeXform_defined(false)
    , flipv_defined(false)
    , subdiv_defined ( false )
    , subframe_defined( false )
    , oid_defined( false )
    , gid_defined( false )
    , pid_defined( false )
    , txvar_defined( false )
    , primvar_defined( false )
    , cycle_defined( false )
    , atJson_defined( false )
    , curveTip_defined( false )
    , curveRoot_defined( false )
    {}
    
    void usage();
    
    //member variables
    std::string filename;
    std::string objectpath;
    double frame;
    double fps;
    double shutterOpen;
    double shutterClose;
    double dt;
    bool excludeXform;
    bool flipv;
    int subdiv;
    int subframe;
    int oid;
    int gid;
    int pid;
    int txvar;
    int primvar;
    int cycle;
    json_object *atJson;
    double curveTip;
    double curveRoot;
    
    std::string getResource( const std::string & name );
    
private:
    
    void applyArgs(ProcArgs & args);
    
    
    bool filename_defined;
    bool objectpath_defined;
    bool frame_defined;
    bool fps_defined;
    bool shutterOpen_defined;
    bool shutterClose_defined;
    bool dt_defined;
    bool excludeXform_defined;
    bool flipv_defined;
    bool subdiv_defined;
    bool subframe_defined;
    bool oid_defined;
    bool gid_defined;
    bool pid_defined;
    bool txvar_defined;
    bool primvar_defined;
    bool cycle_defined;
    bool atJson_defined;
    bool curveTip_defined;
    bool curveRoot_defined;

    
    typedef std::map<std::string, std::string> StringMap;
    typedef boost::shared_ptr<StringMap> StringMapRefPtr;
    typedef std::vector<StringMapRefPtr> StringMapRefPtrVector;
    
    StringMapRefPtrVector resourceSearchPath;
    
    
};

#endif
