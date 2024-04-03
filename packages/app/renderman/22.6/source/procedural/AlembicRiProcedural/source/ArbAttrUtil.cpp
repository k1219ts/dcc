//-*****************************************************************************
//
// Modified for Dexter Pipe-Line
//
// LASTRELEASE
//  - 2017.10.27 $1: AddRlfInjectStructure for MaterialSet dynamic binding
//  - 2018.03.21 $5: Add json overriding for Shape
//                   Fix MatteObject
//
//-*****************************************************************************

#include "ArbAttrUtil.h"
#include <sstream>
#include <cstring>

#include "json.h"

//-*****************************************************************************

ParamListBuilder::~ParamListBuilder()
{
    for ( std::vector<RtString>::iterator I = m_retainedStrings.begin();
            I != m_retainedStrings.end(); ++I )
    {
        free( (*I) );
    }

    m_retainedStrings.clear();
}


//-*****************************************************************************
void ParamListBuilder::add( const std::string & declaration, RtPointer value,
                            ArraySamplePtr sampleToRetain )
{
    //save a copy of the declaration string
    m_retainedStrings.push_back( strdup( declaration.c_str() ) );
    m_outputDeclarations.push_back( m_retainedStrings.back() );

    m_values.push_back( value );

    if ( sampleToRetain )
    {
        m_retainedSamples.push_back( sampleToRetain );
    }
}

//-*****************************************************************************
RtInt ParamListBuilder::n()
{
    return (RtInt) m_values.size();
}

//-*****************************************************************************
RtToken* ParamListBuilder::nms()
{
    if ( m_outputDeclarations.empty() ) { return 0; }

    return &m_outputDeclarations.front();
}

//-*****************************************************************************
RtPointer* ParamListBuilder::vals()
{
    if ( m_values.empty() ) { return NULL; };

    return &m_values.front();
}

//-*****************************************************************************
RtPointer ParamListBuilder::finishStringVector()
{
    RtPointer previous = NULL;

    if ( !m_convertedStringVectors.empty() )
    {
        previous = &( (*m_convertedStringVectors.back())[0] );
    }

    m_convertedStringVectors.push_back( SharedRtStringVector(
            new std::vector<RtString> ) );

    return previous;
}

//-*****************************************************************************
void ParamListBuilder::addStringValue( const std::string &value,
                                            bool retainLocally )
{
    if ( m_convertedStringVectors.empty() )
    {
        finishStringVector();
    }

    if ( retainLocally )
    {
        m_retainedStrings.push_back( strdup( value.c_str() ) );
        m_convertedStringVectors.back()->push_back( m_retainedStrings.back() );
    }
    else
    {
        m_convertedStringVectors.back()->push_back(
            const_cast<RtString>( value.c_str() ) );
    }
}

//-*****************************************************************************
std::string GetPrmanScopeString( GeometryScope scope )
{
    switch (scope)
    {
    case kUniformScope:
        return "uniform";
    case kVaryingScope:
        return "varying";
    case kVertexScope:
        return "vertex";
    case kFacevaryingScope:
        return "facevarying";
    case kConstantScope:
    default:
        return "constant";
    }
}

//-*****************************************************************************
// edit rmantd - move from ArbAttrUtil.h
const std::string GetParamName(std::string name)
{
    boost::regex reg("rman[a]?([c|u|f|v|x])?([F|C|V|N|P|H|M|S])(.*)");
    boost::cmatch m;
    // [0] : full name
    // [1] : c - constant
    //       u - uniform
    //       f - facevarying
    //       v - varying
    //       x - vertex
    // [2] : F - float
    //       C - color
    //       V - vector
    //       N - normal
    //       P - point
    //       H - hpoint
    //       M - matrix
    //       S - string
    // [3] : param name
    if(boost::regex_match(name.c_str(), m, reg))
        return std::string(m[3]);

    return name;
}


void AddStringGeomParamToParamListBuilder(
        ICompoundProperty &parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        ParamListBuilder &paramListBuilder )
{
    IStringGeomParam param( parent, propHeader.getName() );

    if ( !param.valid() )
    {
        //TODO error message?
        return;
    }

    std::string rmanType = "constant string ";

    const std::string &propName = propHeader.getName();

    if( !strncmp(propName.c_str(), "rman__riattr__", 14) )
    {
        std::string::size_type pos = propName.find_last_of( "_" );
        rmanType += propName.substr(pos+1);
    }
    else
    {
        rmanType += propName;
    }

    StringArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();

    paramListBuilder.addStringValue( (*valueSample)[0] );

    RtPointer dataStart = paramListBuilder.finishStringVector();

    paramListBuilder.add(rmanType, dataStart, valueSample);
}


//------------------------------------------------------------------------------
// add rmantd $1
void GetFloatAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, float *value
)
{
    if(!parent.valid()) return;

    const PropertyHeader *header = parent.getPropertyHeader(attributeName);
    if(header)
    {
        const std::string propName = header->getName();
        if(IFloatGeomParam::matches(*header))
        {
            IFloatGeomParam param(parent, propName);
            FloatArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
            if(valueSample->size() > 0) *value = (*valueSample)[0];
        }
    }
}

void GetIntAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, int *value
)
{
    if(!parent.valid()) return;

    const PropertyHeader *header = parent.getPropertyHeader(attributeName);
    if(header)
    {
        const std::string propName = header->getName();
        if(IInt32GeomParam::matches(*header))
        {
            IInt32GeomParam param(parent, propName);
            Int32ArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
            if(valueSample->size() > 0) *value = (*valueSample)[0];
        }
    }
}

void GetStringAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, std::string *value
)
{
    if(!parent.valid()) return;

    const PropertyHeader *header = parent.getPropertyHeader(attributeName);
    if(header)
    {
        const std::string propName = header->getName();
        if(IStringGeomParam::matches(*header))
        {
            IStringGeomParam param(parent, propName);
            StringArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
            if(valueSample->size() > 0) *value = (*valueSample)[0];
        }
    }
}



void getCurveWidthAttributes(ICurvesSchema &cs, ISampleSelector &sampleSelector, float *root, float *tip)
{
    ICompoundProperty geomAttributes = cs.getArbGeomParams();
    if(geomAttributes.valid())
    {
        const std::string baseWidth = "rman__torattr___curveBaseWidth";
        const PropertyHeader *baseWidthHeader = geomAttributes.getPropertyHeader(baseWidth);
        if(baseWidthHeader)
        {
            const std::string baseWidthProp = baseWidthHeader->getName();
            if(IFloatGeomParam::matches(*baseWidthHeader))
            {
                IFloatGeomParam param(geomAttributes, baseWidthProp);
                FloatArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
                if(valueSample->size() > 0) *root = (*valueSample)[0];
            }
        }

        const std::string tipWidth = "rman__torattr___curveTipWidth";
        const PropertyHeader *tipWidthHeader = geomAttributes.getPropertyHeader(tipWidth);
        if(tipWidthHeader)
        {
            const std::string tipWidthProp = tipWidthHeader->getName();
            if(IFloatGeomParam::matches(*tipWidthHeader))
            {
                IFloatGeomParam param(geomAttributes, tipWidthProp);
                FloatArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
                if(valueSample->size() > 0) *tip = (*valueSample)[0];
            }
        }
    }
}


// add_rmantd $1
void AddRlfInjectStructure(ICompoundProperty &parent,
                           ISampleSelector &sampleSelector)
{
    if(!parent.valid()) return;

    std::string mtlSet = "";
    GetStringAttributeValue(parent, sampleSelector, "MaterialSet", &mtlSet);

    if(mtlSet.size() > 0)
    {
        // debug
        // std::cout << "MaterialSet : " << mtlSet << std::endl;

        // base rule
        std::string comment = "RLF Inject SurfaceShading -attribute sets@,initialShadingGroup,";
        comment += mtlSet + ",";

        RiArchiveRecord("structure", (char*) comment.c_str(), RI_NULL);
    }
}

//------------------------------------------------------------------------------
void AddRmanAttributes(
    ICompoundProperty &parent, ISampleSelector &sampleSelector
)
{
    if(!parent.valid()) return;

    for(size_t i=0; i<parent.getNumProperties(); ++i)
    {
        const PropertyHeader &header = parent.getPropertyHeader(i);
        const std::string &propName = header.getName();

        const std::string prefix = propName.substr(0, 14);
        if(!strcmp(prefix.c_str(), "rman__riattr__"))
        {
            std::string suffix = propName.substr(14);
            std::string::size_type pos = suffix.find_first_of(std::string("_"));
            std::string attr = suffix.substr(0,pos);
            std::string param = suffix.substr(pos+1);

            if(strcmp(attr.c_str(), "user"))
            {
                // std::cout << "Attr : " << attr << " Param : " << param << std::endl;
                writeGeomAttribute(parent, sampleSelector, header, propName, attr, param);
            }
        }
    }
}


void AddArbitraryGeomAttributes( IXform &xform,
                                 ICompoundProperty &parent,
                                 ISampleSelector &sampleSelector,
                                 ProcArgs &args )
{
    if( !parent.valid() )
    {
        return;
    }

    for( size_t i = 0; i < parent.getNumProperties(); ++i )
    {
        const PropertyHeader &propHeader = parent.getPropertyHeader(i);
        const std::string &propName = propHeader.getName();

        const std::string rmsPrefix = propName.substr(0,14);

        if( !strcmp(rmsPrefix.c_str(), "rman__riattr__") )
        {
	        std::string rmsSuffix = propName.substr(14);
	        std::string::size_type pos = rmsSuffix.find_first_of( std::string("_") );
	        std::string RiAttrName = rmsSuffix.substr(0,pos);
	        std::string RiParamName = rmsSuffix.substr(pos+1);

	        if( !strcmp(RiParamName.c_str(), "object_id" ) ) {
	            if( args.oid == 1 && args.pid == 0 )
	                writeGeomAttribute( parent, sampleSelector, propHeader, propName, RiAttrName, RiParamName );
	        }
	        else if( !strcmp(RiParamName.c_str(), "group_id" ) ) {
	            if( args.gid == 1 && args.pid == 0 )
	                writeGeomAttribute( parent, sampleSelector, propHeader, propName, RiAttrName, RiParamName );
	        }
	        else if( !strcmp(RiParamName.c_str(), "txVarNum") ) {
	            if( args.txvar == 1 )
	                writeGeomAttribute( parent, sampleSelector, propHeader, propName, RiAttrName, RiParamName );
	        }
	        else if( !strcmp(RiParamName.c_str(), "MatteObject") )
            {
                IInt32GeomParam param( parent, propName );
                Int32ArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
                RiMatte((RtBoolean)(*valueSample)[0]);
            }
            else
                writeGeomAttribute( parent, sampleSelector, propHeader, propName, RiAttrName, RiParamName );
        }
    }
}

//------------------------------------------------------------------------------
// charles edited $5
void AddArbitraryGeomAttributes_fromJson( ICompoundProperty &parent,
                                          ProcArgs &args,
                                          json_object *jsnObj )
{
    json_object_object_foreach(jsnObj, key, jsn)
    {
        const std::string &propName = key;
        const std::string rmsPrefix = propName.substr(0,14);

        if( !strcmp(rmsPrefix.substr(0,14).c_str(), "rman__riattr__") )
        {
            std::string rmsSuffix = propName.substr(14);
            std::string::size_type pos = rmsSuffix.find_first_of( std::string("_") );
            std::string RiAttrName = rmsSuffix.substr(0,pos);
            std::string RiParamName = rmsSuffix.substr(pos+1);

            if( !strcmp(RiParamName.c_str(), "object_id" ) ) {
                if( args.oid == 1 && args.pid == 0 )
                    writeGeomAttribute_fromJson( RiAttrName, RiParamName, jsn);
            }
            else if( !strcmp(RiParamName.c_str(), "group_id" ) ) {
                if( args.gid == 1 && args.pid == 0 )
                    writeGeomAttribute_fromJson( RiAttrName, RiParamName, jsn);
            }
            else if( !strcmp(RiParamName.c_str(), "txVarNum") ) {
                if( args.txvar == 1 )
                    writeGeomAttribute_fromJson( RiAttrName, RiParamName, jsn);
            }
	        else if( !strcmp(RiParamName.c_str(), "MatteObject") )
            {
                RiMatte(json_object_get_boolean(jsn));
            }
            else
                writeGeomAttribute_fromJson( RiAttrName, RiParamName, jsn);
        }
    }
}

void writeGeomAttribute( ICompoundProperty &parent,
                         ISampleSelector &sampleSelector,
                         const PropertyHeader &propHeader,
                         const std::string &propName,
                         std::string RiAttrName,
                         std::string RiParamName)
{
    if( IFloatGeomParam::matches(propHeader) )
    {
        IFloatGeomParam param( parent, propName );
        FloatArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
        if( valueSample->size() > 0 )
        {
            RtFloat floatValue = (RtFloat)(*valueSample)[0];
            std::string parameter = "float ";
            parameter.append( RiParamName );
            RtToken tokens[] = { (char*)parameter.c_str(), 0 };
            RtPointer values[] = { &floatValue, 0 };
            RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
        }
    }
    else if( IInt32GeomParam::matches(propHeader) )
    {
        IInt32GeomParam param( parent, propName );
        Int32ArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
        if( valueSample->size() > 0 )
        {
            RtInt intValue = (RtInt)(*valueSample)[0];
            std::string parameter = "int ";
            parameter.append( RiParamName );
            RtToken tokens[] = { (char*)parameter.c_str(), 0 };
            RtPointer values[] = { &intValue, 0 };
            RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
        }
    }
    else if( IStringGeomParam::matches(propHeader) )
    {
        IStringGeomParam param( parent, propName );
        StringArraySamplePtr valueSample = param.getExpandedValue(sampleSelector).getVals();
        if( valueSample->size() > 0 )
        {
            std::string stringValue = (*valueSample)[0];
            const char * stValue[] = {stringValue.c_str()};
            std::string parameter = "string ";
            parameter.append( RiParamName );
            RtToken tokens[] = { (char*)parameter.c_str(), 0 };
            RtPointer values[] = { stValue, 0 };
            RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
        }
    }
}


// charles edited
void writeGeomAttribute_fromJson( std::string RiAttrName,
                                  std::string RiParamName,
                                  json_object *jsn)
{

    json_type type = json_object_get_type(jsn);

    if( type == json_type_double )
    {
        double value = json_object_get_double(jsn);

        RtFloat floatValue = (RtFloat)(value);
        std::string parameter = "float ";
        parameter.append( RiParamName );
        RtToken tokens[] = { (char*)parameter.c_str(), 0 };
        RtPointer values[] = { &floatValue, 0 };
        RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
    }
    else if( type == json_type_int )
    {
        int32_t value = json_object_get_int(jsn);
        //std::cout << RiParamName << " : " << value << std::endl;
        RtInt intValue = (RtInt)(value);
        std::string parameter = "int ";
        parameter.append( RiParamName );
        RtToken tokens[] = { (char*)parameter.c_str(), 0 };
        RtPointer values[] = { &intValue, 0 };
        RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
    }
    else if( type == json_type_string )
    {
        const char *value = json_object_get_string(jsn);

        const char * stValue[] = {value};
        std::string parameter = "string ";
        parameter.append( RiParamName );
        RtToken tokens[] = { (char*)parameter.c_str(), 0 };
        RtPointer values[] = { stValue, 0 };
        RiAttributeV( (char*)RiAttrName.c_str(), 1, tokens, values );
    }
}

void AddRmanAttributes_fromJson( ProcArgs &args,
                                 json_object *jsnObj )
{
    json_object *jsnAttr;

    // set camera visibility
    if(json_object_object_get_ex(jsnObj, "primaryVisibility", &jsnAttr))
    {
        RtInt camera         = json_object_get_boolean(jsnAttr);

        RiAttribute("visibility", "camera", (RtPointer)&camera, RI_NULL);
    }

    // set transmission visibility
    if(json_object_object_get_ex(jsnObj, "castsShadows", &jsnAttr))
    {
        RtInt transmission   = json_object_get_boolean(jsnAttr);

        RiAttribute("visibility", "transmission", (RtPointer)&transmission, RI_NULL);
    }

    // set holdOut
    if(json_object_object_get_ex(jsnObj, "holdOut", &jsnAttr))
    {
        RtInt holdOut = json_object_get_boolean(jsnAttr);
        RiAttribute("trace", "int holdout", (RtPointer)&holdOut, RI_NULL);
    }

    // set indirect visibility
    bool visRfl = true;
    bool visRfr = true;

    if(json_object_object_get_ex(jsnObj, "visibleInReflections", &jsnAttr))
    {
        visRfl         = json_object_get_boolean(jsnAttr);
    }

    if(json_object_object_get_ex(jsnObj, "visibleInRefractions", &jsnAttr))
    {
        visRfr         = json_object_get_boolean(jsnAttr);
    }

    if(!(visRfl or visRfr))
    {
        RtInt indirect = 0;
        RiAttribute("visibility", "indirect", (RtPointer)&indirect, RI_NULL);
    }

    // set doubleSide and opposite
    if(json_object_object_get_ex(jsnObj, "doubleSided", &jsnAttr))
    {
        RiSides(json_object_get_boolean(jsnAttr) + 1);
    }

    if(json_object_object_get_ex(jsnObj, "opposite", &jsnAttr))
    {
        if(json_object_get_boolean(jsnAttr))
            RiOrientation("rh");
        else
            RiOrientation("lh");
    }

    json_object_object_foreach(jsnObj, key, jsn)
    {
        const std::string &propName = key;
        const std::string rmsPrefix = propName.substr(0,14);

        if( !strcmp(rmsPrefix.substr(0,14).c_str(), "rman__riattr__") )
        {
            std::string rmsSuffix = propName.substr(14);
            std::string::size_type pos = rmsSuffix.find_first_of( std::string("_") );
            std::string RiAttrName = rmsSuffix.substr(0,pos);
            std::string RiParamName = rmsSuffix.substr(pos+1);

            if(strcmp(RiAttrName.c_str(), "user"))
            {
                // std::cout << "Attr : " << attr << " Param : " << param << std::endl;
                writeGeomAttribute_fromJson( RiAttrName, RiParamName, jsn);
            }
        }
    }
}
// end


//-*****************************************************************************
void AddArbitraryGeomParams( ICompoundProperty &parent,
                             ISampleSelector &sampleSelector,
                             ParamListBuilder &paramListBuilder,
                             const std::set<std::string> * excludeNames
                           )
{
    if ( !parent.valid() )
    {
        return;
    }

    //std::cout << "_______________________________________________" << std::endl;
    //std::cout << "Shape Name >> " << parent.getObject().getName() << std::endl;

    for ( size_t i = 0; i < parent.getNumProperties(); ++i )
    {
        const PropertyHeader &propHeader = parent.getPropertyHeader( i );
        const std::string &propName = propHeader.getName();

        if (propName.empty()
            || ( excludeNames
                 && excludeNames->find( propName ) != excludeNames->end() ) )
        {
            continue;
        }
        // debug
        //std::cout << "primvar >> ";

        AddArbitraryGeomParam( parent, sampleSelector, paramListBuilder, propHeader );
    }
}

void AddArbitraryGeomParam( ICompoundProperty &parent,
                            ISampleSelector &sampleSelector,
                            ParamListBuilder &paramListBuilder,
                            const PropertyHeader &propHeader )
{
    if ( IFloatGeomParam::matches( propHeader ) )
    {
        //std::cout << "IFloatGeomParam";
        AddGeomParamToParamListBuilder<IFloatGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "float",
            paramListBuilder);
    }
    else if ( IDoubleGeomParam::matches( propHeader ) )
    {
        //std::cout << "IDoubleGeomParam";
        AddGeomParamToParamListBuilderAsFloat<IDoubleGeomParam, double>(
            parent,
            propHeader,
            sampleSelector,
            "float",
            paramListBuilder);
    }
    else if ( IV3dGeomParam::matches( propHeader ) )
    {
        //std::cout << "IV3dGeomParam";
        AddGeomParamToParamListBuilderAsFloat<IV3dGeomParam, double>(
            parent,
            propHeader,
            sampleSelector,
            "vector",
            paramListBuilder);
    }
    else if ( IInt32GeomParam::matches( propHeader ) )
    {
        //std::cout << "IInt32GeomParam";
        AddGeomParamToParamListBuilder<IInt32GeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "int",
            paramListBuilder);
    }
    else if ( IStringGeomParam::matches( propHeader ) )
    {
        //std::cout << "IStringGeomParam";
        AddStringGeomParamToParamListBuilder(
                parent, propHeader, sampleSelector, paramListBuilder);
    }
    else if ( IV2fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IV2fGeomParam";
        AddGeomParamToParamListBuilder<IV2fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "float",
            paramListBuilder,
            2);
    }
    else if ( IV3fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IV3fGeomParam";
        AddGeomParamToParamListBuilder<IV3fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "vector",
            paramListBuilder);
    }
    else if ( IP3fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IP3fGeomParam";
        AddGeomParamToParamListBuilder<IP3fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "point",
            paramListBuilder);
    }
    else if ( IP3dGeomParam::matches( propHeader ) )
    {
        //std::cout << "IP3dGeomParam";
        AddGeomParamToParamListBuilderAsFloat<IP3dGeomParam, double>(
            parent,
            propHeader,
            sampleSelector,
            "point",
            paramListBuilder);
    }
    else if ( IN3fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IN3fGeomParam";
        AddGeomParamToParamListBuilder<IN3fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "normal",
            paramListBuilder);
    }
    else if ( IC3fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IC3fGeomParam";
        AddGeomParamToParamListBuilder<IC3fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "color",
            paramListBuilder);
    }
    else if ( IM44fGeomParam::matches( propHeader ) )
    {
        //std::cout << "IM44fGeomParam";
        AddGeomParamToParamListBuilder<IM44fGeomParam>(
            parent,
            propHeader,
            sampleSelector,
            "matrix",
            paramListBuilder);
    }
    else if ( IBoolGeomParam::matches( propHeader ) )
    {
        //std::cout << "IBoolGeomParam";
        AddGeomParamToParamListBuilderAsInt<IBoolGeomParam, bool_t>(
            parent,
            propHeader,
            sampleSelector,
            paramListBuilder);
    }
}

// charles edited $5
void AddArbitraryGeomParams_fromJson( ICompoundProperty & parent,
                                      ISampleSelector &sampleSelector,
                                      json_object *jsnObj,
                                      ParamListBuilder &paramListBuilder,
                                      const std::set<std::string> * excludeNames )
{

    //std::cout << "______________JSON_________________JSON________________" << std::endl;
    //std::cout << "Shape Name >> " << parent.getObject().getName() << std::endl;

    json_object_object_foreach(jsnObj, key, jsn)
    {
        if( strncmp(key, "rman", 4) ||
            !strncmp(key, "rman__torattr", 13) )
            continue;

        if( !strncmp(key, "rman__riattr", 12) )
        {
            std::string rmanType  = "constant ";
            std::string paramName(key);
            std::string::size_type pos = paramName.find_last_of( "_" );
            paramName  = paramName.substr(pos+1);

            switch(json_object_get_type(jsn))
            {
                case json_type_int:
                {
                    rmanType += "int " + paramName;
                    RtInt val = json_object_get_int(jsn);
                    paramListBuilder.addAsInt(rmanType, &val, 1);
                }
                break;
                case json_type_double:
                {
                    rmanType += "float " + paramName;
                    RtFloat val = json_object_get_double(jsn);
                    paramListBuilder.addAsFloat(rmanType, &val, 1);
                }
                break;
                case json_type_string:
                {
                    rmanType += "string " + paramName;
                    std::string val(json_object_get_string(jsn));

                    paramListBuilder.addStringValue( val, true );

                    RtPointer dataStart = paramListBuilder.finishStringVector();
                    paramListBuilder.add(rmanType, dataStart );
                }
                break;
            }
        }
        else if( !strncmp(key, "rman", 4) )
        {
            // in this case, I supposed the attribute is shell-id.
            // That's why, the attribute must be in the alembic too.
            if ( !parent.valid() )
            {
                return;
            }

            const PropertyHeader *propHeader = parent.getPropertyHeader( std::string(key) );
            const std::string &propName = propHeader->getName();

            if (propName.empty()
                || ( excludeNames
                     && excludeNames->find( propName ) != excludeNames->end() ) )
            {
                continue;
            }

            AddArbitraryGeomParam( parent, sampleSelector, paramListBuilder, *propHeader );
        }
    }
}
