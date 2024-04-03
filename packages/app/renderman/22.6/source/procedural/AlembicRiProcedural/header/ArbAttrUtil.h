//  rman.td edit
//      2016.03.06 $4
#ifndef _Alembic_Prman_ArbAttrUtil_h_
#define _Alembic_Prman_ArbAttrUtil_h_

#include <ri.h>

#include <Alembic/AbcGeom/All.h>
#include <boost/smart_ptr.hpp>
#include <set>
#include <regex>
#include "ProcArgs.h"
#include <boost/regex.hpp>

#include "json.h"

using namespace Alembic::AbcGeom;

//-*****************************************************************************
class ParamListBuilder
{
public:

    ~ParamListBuilder();

    void add( const std::string & declaration, RtPointer value,
              ArraySamplePtr sampleToRetain = ArraySamplePtr() );

    // used for conversion of DoubleGeomParams
    // NOTE: In addition to performing the conversion, this also makes and
    // returns an internal copy of the data. Even in cases in which you're
    // starting with float data, this can be usful if you need to manipulate
    // the values after-the-fact.
    template <typename T>
    std::vector<RtFloat> * addAsFloat( const std::string & declaration, const T * length,
            size_t numValues );

    template <typename T>
    std::vector<RtInt> * addAsInt( const std::string & declaration, const T * length,
            size_t numValues );

    //returns the start of the current vector and pushes a new one
    RtPointer finishStringVector();

    //If retainLocally is false, it's expected that "value" remain alive
    //The common case is out of a StringArraySamplePtr which is provided
    //as sampleToRetain in the add method
    void addStringValue( const std::string & value,
                              bool retainLocally = false );

    RtInt n();
    RtToken* nms();
    RtPointer* vals();

private:
    std::vector<RtToken> m_outputDeclarations;
    std::vector<RtPointer> m_values;
    std::vector<ArraySamplePtr> m_retainedSamples;

    std::vector<RtString> m_retainedStrings;

    typedef boost::shared_ptr<std::vector<RtString> > SharedRtStringVector;
    std::vector<SharedRtStringVector> m_convertedStringVectors;

    typedef boost::shared_ptr<std::vector<RtFloat> > SharedRtFloatVector;
    std::vector<SharedRtFloatVector> m_convertedFloatVectors;

    typedef boost::shared_ptr<std::vector<RtInt> > SharedRtIntVector;
    std::vector<SharedRtIntVector> m_convertedIntVectors;

};

//-*****************************************************************************

template <typename T>
std::vector<RtFloat> * ParamListBuilder::addAsFloat( const std::string & declaration,
        const T * value, size_t length)
{
    SharedRtFloatVector convertedValues( new std::vector<RtFloat> );
    convertedValues->reserve( length );

    convertedValues->insert( convertedValues->end(), value, value + length );
    m_convertedFloatVectors.push_back( convertedValues );

    add( declaration, &( ( *convertedValues )[0] ) );

    return &(*convertedValues);
}

//-*****************************************************************************

template <typename T>
std::vector<RtInt> * ParamListBuilder::addAsInt( const std::string & declaration,
        const T * value, size_t length)
{
    SharedRtIntVector convertedValues( new std::vector<RtInt> );
    convertedValues->reserve( length );

    convertedValues->insert( convertedValues->end(), value, value + length );
    m_convertedIntVectors.push_back( convertedValues );

    add( declaration, &( ( *convertedValues )[0] ) );

    return &(*convertedValues);
}



//-*****************************************************************************
std::string GetPrmanScopeString( GeometryScope scope );

//-*****************************************************************************
template <typename T>
void AddGeomParamToParamListBuilder( ICompoundProperty & parent,
                                             const PropertyHeader &propHeader,
                                             ISampleSelector &sampleSelector,
                                             const std::string &rmanBaseType,
                                             ParamListBuilder &ParamListBuilder,
                                             size_t baseArrayExtent = 1,
                                             const std::string & overrideName = "" )
{
    T param( parent, propHeader.getName() );

    if ( !param.valid() )
    {
        //TODO error message?
        return;
    }

    const std::string &propName = propHeader.getName();
    std::string rmanType;
    if( !strncmp(propName.c_str(), "rman__riattr", 12) )
    {
        rmanType += "constant " + rmanBaseType + " ";

        std::string::size_type pos = propName.find_last_of( "_" );

        rmanType += propName.substr(pos+1);
    }
    else if( !strncmp(propName.c_str(), "rman__torattr", 13) )
    {
        return;
    }
    else
    {
        rmanType  = GetPrmanScopeString( param.getScope() ) + " ";
        rmanType += rmanBaseType;

        size_t arrayExtent = baseArrayExtent * param.getArrayExtent();
        if( arrayExtent > 1 )
        {
            std::ostringstream buffer;
            buffer << "[" << arrayExtent << "]";
            rmanType += buffer.str();
        }

        rmanType += " " + ( overrideName.empty() ? propHeader.getName() : overrideName );
    }

    typename T::prop_type::sample_ptr_type valueSample =
            param.getExpandedValue( sampleSelector ).getVals();

    ParamListBuilder.add( rmanType, (RtPointer)valueSample->get(),
            valueSample );
}
//-*****************************************************************************


const std::string GetParamName(std::string name);

// NOTE: In addition to performing the conversion, this also makes and
// returns an internal copy of the data. Even in cases in which you're
// starting with float data, this can be usful if you need to manipulate
// the values after-the-fact. The AddGeomParamToParamListBuilder codepath
// normally avoids a copy of the data as the GeomParam samples can be handed
// to RenderMan directly.
template <typename T, typename podT>
std::vector<RtFloat> * AddGeomParamToParamListBuilderAsFloat( ICompoundProperty & parent,
                                             const PropertyHeader &propHeader,
                                             ISampleSelector &sampleSelector,
                                             const std::string &rmanBaseType,
                                             ParamListBuilder &ParamListBuilder,
                                             const std::string & overrideName = ""
                                           )
{
    T param( parent, propHeader.getName() );

    if ( !param.valid() )
    {
        //TODO error message?
        return 0;
    }

    std::string rmanType = GetPrmanScopeString(param.getScope()) + " ";
    rmanType += rmanBaseType;

    size_t dataTypeExtent = T::prop_type::traits_type::dataType().getExtent();

    // If rmanBaseType is something other than float, the assumption is that the
    // specified rman type is in agreement with the dataTypeExtent -- i.e. color = 3
    // If it's just float, we want to make sure that our data footprint matches
    // that of the incoming type in RenderMan terms.
    size_t baseArrayExtent = (rmanBaseType == "float") ? dataTypeExtent: 1;

    size_t arrayExtent = baseArrayExtent * param.getArrayExtent();
    if (arrayExtent > 1)
    {
        std::ostringstream buffer;
        buffer << "[" << arrayExtent << "]";
        rmanType += buffer.str();
    }

    if( overrideName.empty() )
        rmanType += " " + GetParamName(propHeader.getName());
    else
        rmanType = overrideName;

    // debug
    //std::cout << rmanType << std::endl;

    typename T::prop_type::sample_ptr_type valueSample =
            param.getExpandedValue( sampleSelector ).getVals();


    //always grab the natural extent of the incoming data for the internal copy
    int rawExtent = dataTypeExtent * param.getArrayExtent();

    return ParamListBuilder.addAsFloat( rmanType,
            reinterpret_cast<const podT *>( valueSample->get() ),
                    valueSample->size() * rawExtent );

}

//-*****************************************************************************

// NOTE: In addition to performing the conversion, this also makes and
// returns an internal copy of the data. Even in cases in which you're
// starting with int data, this can be usful if you need to manipulate
// the values after-the-fact. The AddGeomParamToParamListBuilder codepath
// normally avoids a copy of the data as the GeomParam samples can be handed
// to RenderMan directly.
template <typename T, typename podT>
std::vector<RtInt> * AddGeomParamToParamListBuilderAsInt( ICompoundProperty & parent,
                                             const PropertyHeader &propHeader,
                                             ISampleSelector &sampleSelector,
                                             ParamListBuilder &ParamListBuilder,
                                             const std::string & overrideName = ""
                                           )
{
    T param( parent, propHeader.getName() );

    if ( !param.valid() )
    {
        //TODO error message?
        return 0;
    }

    //Unlike RtFloat, "int" is the only RenderMan type that takes something
    //compatible with RtInt. We can assume anything sent here is intended to be
    //written to RenderMan as int[n].
    const std::string rmanBaseType("int");

    std::string rmanType = GetPrmanScopeString( param.getScope() ) + " ";

    rmanType += rmanBaseType;

    size_t dataTypeExtent = T::prop_type::traits_type::dataType().getExtent();



    size_t arrayExtent = dataTypeExtent * param.getArrayExtent();
    if (arrayExtent > 1)
    {
        std::ostringstream buffer;
        buffer << "[" << arrayExtent << "]";
        rmanType += buffer.str();
    }

    rmanType += " " + (
            overrideName.empty() ? propHeader.getName() : overrideName );


    typename T::prop_type::sample_ptr_type valueSample =
            param.getExpandedValue( sampleSelector ).getVals();


    return ParamListBuilder.addAsInt( rmanType,
            reinterpret_cast<const podT *>( valueSample->get() ),
                    valueSample->size() * arrayExtent );

}




//-*****************************************************************************
// add_rmantd editing
void GetFloatAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, float *value
);

void GetIntAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, int *value
);

void GetStringAttributeValue(
    ICompoundProperty &parent, ISampleSelector &sampleSelector,
    std::string attributeName, std::string *value
);

void getCurveWidthAttributes(
    ICurvesSchema &cs, ISampleSelector &sampleSelector,
    float *root, float *tip);

void AddRlfInjectStructure(
    ICompoundProperty &parent, ISampleSelector &sampleSelector
);

void AddRmanAttributes(
    ICompoundProperty &parent, ISampleSelector &sampleSelector
);

void AddRmanAttributes_fromJson( ProcArgs &args,
                                  json_object *jsnObj );

void AddArbitraryGeomAttributes( IXform &xform,
                                 ICompoundProperty &parent,
                                 ISampleSelector &sampleSelector,
                                 ProcArgs &args );

void AddArbitraryGeomAttributes_fromJson( ICompoundProperty &parent,
                                          ProcArgs &args,
                                          json_object *jsnObj );

void writeGeomAttribute( ICompoundProperty &parent,
                         ISampleSelector &sampleSelector,
                         const PropertyHeader &propHeader,
                         const std::string &propName,
                         std::string RiAttrName,
                         std::string RiParamName );

// charles added
void writeGeomAttribute_fromJson( std::string RiAttrName,
                                  std::string RiParamName,
                                  json_object *jsn);




//-*****************************************************************************

void AddArbitraryGeomParams( ICompoundProperty &parent,
                             ISampleSelector &sampleSelector,
                             ParamListBuilder &ParamListBuilder,
                             const std::set<std::string> * excludeNames = NULL );

void AddArbitraryGeomParam( ICompoundProperty &parent,
                           ISampleSelector &sampleSelector,
                           ParamListBuilder &paramListBuilder,
                           const PropertyHeader &propHeader );

//-*****************************************************************************
void AddArbitraryGeomParams_fromJson( ICompoundProperty & parent,
                                      ISampleSelector &sampleSelector,
                                      json_object *jsnObj,
                                      ParamListBuilder &paramListBuilder,
                                      const std::set<std::string> * excludeNames = NULL );

//[array, scalar], [as float, as int, as string];


template <typename propT, typename podT>
void AddArrayPropertyAsFloatToParamListBuilder(
        ICompoundProperty & parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        const std::string & name,
        const std::string & rmanBaseType,
        ParamListBuilder & paramListBuilder)
{
    propT prop(parent, propHeader.getName());

    size_t dataTypeExtent = propT::traits_type::dataType().getExtent();

    typename propT::sample_ptr_type sample = prop.getValue(sampleSelector);

    std::ostringstream buffer;
    buffer << rmanBaseType;

    size_t baseArrayExtent = (rmanBaseType == "float") ? dataTypeExtent: 1;
    size_t arrayExtent = baseArrayExtent * sample->size();
    if (arrayExtent > 1)
    {
        buffer << "[" << arrayExtent << "]";
    }

    buffer << " " << name;

    paramListBuilder.addAsFloat<podT>(
            buffer.str(), reinterpret_cast<const podT *>( sample->get() ),
                    sample->size() * dataTypeExtent);
}


template <typename propT, typename podT>
void AddScalarPropertyAsFloatToParamListBuilder(
        ICompoundProperty & parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        const std::string & name,
        const std::string & rmanBaseType,
        ParamListBuilder &paramListBuilder)
{
    propT prop(parent, propHeader.getName());

    typename propT::value_type scalarValue;
    prop.get(scalarValue, sampleSelector);

    std::ostringstream buffer;
    buffer << rmanBaseType;

    size_t dataTypeExtent =
            propT::traits_type::dataType().getExtent();

    if (rmanBaseType == "float" && dataTypeExtent > 1)
    {
        buffer << "[" << dataTypeExtent << "] ";
    }
    buffer << " " << name;

    paramListBuilder.addAsFloat<podT>(
            buffer.str(), reinterpret_cast<const podT *>(&scalarValue),
                    dataTypeExtent);
}


// template <typename propT, typename podT>
// void AddArrayPropertyAsIntToParamListBuilder(
//         ICompoundProperty & parent,
//         const PropertyHeader &propHeader,
//         ISampleSelector &sampleSelector,
//         const std::string & name,
//         ParamListBuilder &ParamListBuilder);

template <typename propT, typename podT>
void AddScalarPropertyAsIntToParamListBuilder(
        ICompoundProperty & parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        const std::string & name,
        ParamListBuilder &paramListBuilder)
{
    propT prop(parent, propHeader.getName());


    typename propT::value_type scalarValue;
    prop.get(scalarValue, sampleSelector);

    std::ostringstream buffer;
    buffer << "int";

    size_t dataTypeExtent =
            propT::traits_type::dataType().getExtent();

    if (dataTypeExtent > 1)
    {
        buffer << "[" << dataTypeExtent << "]";
    }
    buffer << " " << name;

    paramListBuilder.addAsInt<podT>(
            buffer.str(), reinterpret_cast<const podT *>(&scalarValue),
                    dataTypeExtent);
}

template <typename propT>
void AddArrayPropertyAsStringToParamListBuilder(
        ICompoundProperty & parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        const std::string & name,
        ParamListBuilder &paramListBuilder)
{
    propT prop(parent, propHeader.getName());


    StringArraySamplePtr valueSample = prop.getValue(sampleSelector);

    std::ostringstream buffer;
    buffer << "string";

    if (valueSample->size() > 1)
    {
        buffer << "[" << valueSample->size() << "]";
    }

    buffer << " " << name;

    for ( size_t i = 0; i < valueSample->size(); ++i )
    {
        paramListBuilder.addStringValue( (*valueSample)[i] );
    }

    RtPointer dataStart = paramListBuilder.finishStringVector();

    paramListBuilder.add(buffer.str(), dataStart, valueSample);
}



template <typename propT>
void AddScalarPropertyAsStringToParamListBuilder(
        ICompoundProperty & parent,
        const PropertyHeader &propHeader,
        ISampleSelector &sampleSelector,
        const std::string & name,
        ParamListBuilder &paramListBuilder)
{
    propT prop(parent, propHeader.getName());

    typename propT::value_type scalarValue;
    prop.get(scalarValue, sampleSelector);

    std::ostringstream buffer;
    buffer << "string";

    size_t dataTypeExtent =
            propT::traits_type::dataType().getExtent();

    if (dataTypeExtent > 1)
    {
        buffer << "[" << dataTypeExtent << "]";
    }
    buffer << " " << name;


    for (size_t i = 0; i < dataTypeExtent; ++i)
    {
        paramListBuilder.addStringValue(
                reinterpret_cast<std::string*>(&scalarValue)[i], true);
    }

    paramListBuilder.add(buffer.str(),
            paramListBuilder.finishStringVector());
}




#endif
