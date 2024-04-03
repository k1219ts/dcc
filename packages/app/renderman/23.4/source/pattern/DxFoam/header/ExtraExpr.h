#ifndef ExtraExpr_h
#define ExtraExpr_h

#include "SeExpr2/ExprBuiltins.h"
#include "SeExpr2/ExprNode.h"

namespace SeExpr2 {

struct VoronoiPointData : public ExprFuncNode::Data
{
    Vec3d points[27];
    Vec3d cell;
    double jitter;
    VoronoiPointData() : jitter(-1) {}
};
Vec3d voronoiFn(VoronoiPointData& data, int n, const Vec3d* args);
Vec3d cvoronoiFn(VoronoiPointData& data, int n, const Vec3d* args);
Vec3d pvoronoiFn(VoronoiPointData& data, int n, const Vec3d* args);

}

#endif
