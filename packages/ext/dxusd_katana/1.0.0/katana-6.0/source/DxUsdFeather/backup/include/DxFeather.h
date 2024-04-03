#pragma once

#include "DxPoint.h"
#include "DxVector.h"
#include "DxMatrix.h"
#include "DxFloat2.h"
#include "DxNurbsCurve.h"

namespace Dx
{
    Dx::Matrix axisToMatrix(const Dx::Point& T, Dx::Point RAx, Dx::Point RAy, Dx::Point RAz, float s)
    {
        RAx.normalize();
        RAy.normalize(); // T
        RAz.normalize(); // U

        // std::cout << s << std::endl;

        Dx::Matrix M = Dx::Matrix(s * RAx.x, s * RAx.y, s * RAx.z, 0,
                                  s * RAy.x, s * RAy.y, s * RAy.z, 0,
                                  s * RAz.x, s * RAz.y, s * RAz.z, 0,
                                  T.x,       T.y,       T.z,       1);
        return M;
    }

    class Feather
    {
    public:
        Dx::NurbsCurve m_ncRachis;
        Dx::NurbsCurve m_ncEdges[2]; // Left Right

        Dx::Matrix M;
        Dx::Matrix IM;
        float m_fScale;

    public:
        Feather()
        {

        }
        Feather(const std::vector<Dx::Point> cvs)
        {
            M           = Dx::Matrix();
            IM          = Dx::Matrix();

            m_fScale = 1.0f;

            setOutlines(cvs);
        }

        void setOutlines(const std::vector<Dx::Point> cvs)
        {
            const int n = cvs.size() / 3;
            std::vector<int> r = std::vector<int>();
            std::vector<int> e[2] = {std::vector<int>(), std::vector<int>()};

            std::vector<Dx::Point> rachisCvs = std::vector<Dx::Point>();
            std::vector<Dx::Point> edges0Cvs = std::vector<Dx::Point>();
            std::vector<Dx::Point> edges1Cvs = std::vector<Dx::Point>();

            for(int i = n; i < 2*n; ++i)
            {
                r.push_back(i);
                rachisCvs.push_back(cvs[i]);
            }

            m_ncRachis = Dx::NurbsCurve(rachisCvs);

            for(int i = 2 * n; i < 3 * n; ++i)
            {
                e[1].push_back(i);
            }
            for(int j = 0; j < n; ++j)
            {
                e[0].push_back(j);
            }
            for(int s = 0; s < 2; ++s)
            {
                e[s].insert(e[s].begin(), 1, r[0]);
                e[s].push_back(r.back());
            }

            for(int i: e[0])
            {
                edges0Cvs.push_back(cvs[i]);
            }
            for(int i: e[1])
            {
                edges1Cvs.push_back(cvs[i]);
            }

            m_ncEdges[0] = Dx::NurbsCurve(edges0Cvs);
            m_ncEdges[1] = Dx::NurbsCurve(edges1Cvs);
        }

        /*
         @parm
            rp  :   Dx::Point  @ rachis Parameter
            r   :   Dx::Point  @ rachis Position
            F   :   Dx::Point  @ rachis Front Vector
         @comment
            Vector3f * Matrix4f = Vector3f
        */
        Dx::Matrix barbTangentSpace(const float& rp, const Dx::Point& r, const Dx::Point& F)
        {
            float edgeRachisParam = rp / m_ncRachis.m_vKnotList.back() * m_ncEdges[0].m_vKnotList.back();

            Dx::Point rachisIdentityMatrix = r * IM;
            Dx::Vector frontVector = Dx::Vector(F.x, F.y, F.z) * IM; // df.Vector(F)

            float min_param = m_ncEdges[0].m_vKnotList.back() - 0.001f;
            if(edgeRachisParam >= min_param)
                edgeRachisParam = min_param;


            m_ncEdges[0].position(edgeRachisParam);
            m_ncEdges[1].position(edgeRachisParam);

            Dx::Point ep0 = m_ncEdges[0].m_vPos[0] * IM - rachisIdentityMatrix;
            Dx::Point ep1 = m_ncEdges[1].m_vPos[0] * IM - rachisIdentityMatrix;

            ep0.normalize();
            ep1.normalize();

            Dx::Vector T = Dx::Vector(); // dt.Vector
            Dx::Vector U = Dx::Vector(); // dt.Vector

            if (ep0.z == 0)
                T = Dx::Vector(-ep0.x, ep0.y, ep0.z); // dt.Vector
            else if (ep1.z == 0)
                T = Dx::Vector(ep1.x, ep1.y, ep1.z); // dt.Vector
            else
            {
                Dx::Point _T = ep1 / fabs(sin(ep1.z)) - ep0 / fabs(sin(ep0.z));
                T = Dx::Vector(_T.x, _T.y, _T.z);
            }

            U = frontVector.cross(T);
            T = U.cross(frontVector);

            // // $1
            // Dx::Point _U = Dx::Point(U.x, U.y, U.z);
            // Dx::Point _T = Dx::Point(T.x, T.y, T.z);
            //
            // _U.normalize();
            // _T.normalize();
            //
            // if(isnan(_U.x) || isnan(_U.y) || isnan(_U.z))
            // {
            //     std::cout << "###############################################" << std::endl;
            //     std::cout << "edgeRachisParam : " << edgeRachisParam << std::endl;
            //     ep0.Print();
            //     ep1.Print();
            // }
            // // $1 end

            Dx::Matrix TS = axisToMatrix(r, Dx::Point(frontVector.x, frontVector.y, frontVector.z), Dx::Point(T.x, T.y, T.z), Dx::Point(U.x, U.y, U.z), 1.0f);

            return TS;
        }
    };

    class SourceFeather : public Feather
    {
    public:
        /*
        barbCVs     : barb's cvs
                      [[[point, ... ], ... ], [[point, ... ], ...]]             Float3Array Array * 2
        barbParams  : barb's cvs
                      [[[float, float], ... ], [[float, float], ...]]           Float2Array * 2
        barbUParams : barb cvs' parameters on U
                      [[[float, ...], ...], [[float, ...], ...]]                FloatArray Array * 2
        barbDisps   : uparameter to barb cv
                      [[[float, ...], ...], [[float, ...], ...]]                FloatArray Array * 2

        frontParam  : front parameter on root edge cvs
                      (float)
        barb_uInitCVs    : initial cvs on barb_uVectors
                      [[[point, ...], ...], [[point, ...], ...]]                Float3Array Array * 2
        barb_uVectors    : vector param(richis) to param(edge) in tangentSpace
                      [[point, ...], [point, ...]]                              Float3Array * 2

        rachisCVs       : rachis's cvs
                          [point, ...]
        rachisUParams   : rachis cvs' parameters on U
                          [float, ...]
        rachis_uInitCVs : initial cvs on rachis of lamenation
                          [point, ...]
        */

        // Barbs
        std::vector<std::vector<Dx::Point>>    m_vBarbCVs[2];      // CVS
        std::vector<Dx::Float2>                 m_vBarbParams[2];   // RE
        std::vector<std::vector<float>>         m_vBarbUParams[2];  // U

        std::vector<std::vector<Dx::Point>>    m_vBarb_uInitCVs[2];
        std::vector<Dx::Point>                 m_vBarb_uVectors[2];//

        // Rachis
        std::vector<Dx::Point>                 m_vRachisCVs;
        std::vector<float>                      m_vRachisUParams;

        std::vector<Dx::Point>                 m_vRachis_uInitCVs;

        float                                   frontParam;


    public:
        SourceFeather()
        {

        }
        SourceFeather(std::vector<Dx::Point> _cvs)
            : Feather(_cvs)
        {
            frontParam = 0.0f;
        }

        bool setFrontParam()
        {
            Dx::Point center   = m_ncRachis.m_vCvList[0];
            Dx::Point left     = m_ncEdges[0].m_vCvList[1];
            Dx::Point right    = m_ncEdges[1].m_vCvList[1];
            Dx::Point front    = m_ncRachis.m_vCvList[1];

            std::vector<int> d = {0, 1};

            float a1 = right[d[0]] - left[d[0]];
            float a2 = front[d[0]] - center[d[0]];
            float a3 = center[d[0]] - left[d[0]];

            if (a2 == 0)
            {
                if (a1 == 0)
                    return false;
                else
                {
                    frontParam = a3 / a1;
                    return false;
                }
            }

            float b1 = right[d[1]] - left[d[1]];
            float b2 = front[d[1]] - center[d[1]];
            float b3 = center[d[1]] - left[d[1]];

            frontParam = (b2 * a3 / a2 - b3) / (b2 * a1 / a2 - b1);
        }

        void set()
        {
            setFrontParam();

            // set barbs

            for(int s = 0; s < 2; s++) // left : 0, right : 1
            {
                for(int b = 0; b < m_vBarbUParams[s].size(); ++b)
                {
                    float rachisParam   = m_vBarbParams[s][b][0];
                    float edgeParam     = m_vBarbParams[s][b][1];

                    m_ncRachis.position(rachisParam, 1);
                    m_ncEdges[s].position(edgeParam);
                    Dx::Point r = m_ncRachis.m_vPos[0];
                    Dx::Point F = m_ncRachis.m_vPos[1];
                    Dx::Point e = m_ncEdges[s].m_vPos[0];

                    // Set Barb UInit CVS
                    std::vector<Dx::Point> cvs = std::vector<Dx::Point>();
                    for(const float& u : m_vBarbUParams[s][b])
                    {
                        Dx::Point cv = ((e - r) * u) + r;
                        cvs.push_back(cv);
                    }

                    m_vBarb_uInitCVs[s].push_back(cvs);

                    // Set barb uVectors in tangent Space
                    Dx::Matrix TangentSpaceMatrix = barbTangentSpace(rachisParam, r, F);

                    Dx::Point e_TS = e * TangentSpaceMatrix.inverse();

                    m_vBarb_uVectors[s].push_back(e_TS);
                }
            }

            // Set Rachis uInitCVS
            for(const float& u : m_vRachisUParams)
            {
                m_ncRachis.position(u);
                Dx::Point cv = m_ncRachis.m_vPos[0];

                m_vRachis_uInitCVs.push_back(cv);
            }
        }
    };

    class DeformFeather : public Feather
    {
    public:
        SourceFeather       m_SourceFeather;
        float               m_fScale;

    public:
        DeformFeather();
        DeformFeather(const SourceFeather& sourceFeather, std::vector<Dx::Point> _cvs)
            : Feather(_cvs)
        {
            m_SourceFeather = sourceFeather;

            setTransform();
        }

        void setTransform()
        {
            std::vector<Dx::Point> opnts = { m_SourceFeather.m_ncRachis.m_vCvList[0],
                                            m_SourceFeather.m_ncEdges[0].m_vCvList[1],
                                            m_SourceFeather.m_ncEdges[1].m_vCvList[1],
                                            m_SourceFeather.m_ncRachis.m_vCvList[1]
                                            };

            std::vector<Dx::Point> dpnts = { m_ncRachis.m_vCvList[0],
                                                m_ncEdges[0].m_vCvList[1],
                                                m_ncEdges[1].m_vCvList[1],
                                                m_ncRachis.m_vCvList[1]
                                            };

            Dx::Point T = dpnts[0] - opnts[0];
            Dx::Point o = opnts[1] - opnts[2];
            Dx::Point d = dpnts[1] - dpnts[2];

            float s = d.length() / o.length();

            Dx::Point frontVector = (dpnts[2] - dpnts[1]) * m_SourceFeather.frontParam + dpnts[1] - dpnts[0];
            Dx::Point upVector = (dpnts[2] - dpnts[0]).cross(dpnts[1] - dpnts[0]);
            Dx::Point sideVector = frontVector.cross(upVector);

            m_fScale            = s;
            M                   = axisToMatrix(T, sideVector, frontVector, upVector, s);
            IM                  = M.inverse();
        }

        float reparameter(float u, float k)
        {
            float _u = (u < 0) ? 0 : u;
            _u = (_u > 1) ? 1 : _u;

            if (k == 0.5)
                return _u;
            else
            {
                float tmp = 1 - 2 * k;
                float reparm = (sqrt(k * k + _u * tmp) - k) / tmp;

                return (reparm > 0.0f) ? reparm : 0.0f;
            }
        }

        Dx::Point getK(Dx::Point e, Dx::Point u, Dx::Matrix TS)
        {
            // e in tangentSpace
            Dx::Point e_TS = e * TS.inverse();
            e_TS.normalize();

            // angle between u and e
            Dx::Point nu = u;
            nu.normalize();

            float d  = nu.dot(e_TS);
            float th = acosf(d);

            if (isnan(th))
                th = 0.0f;

            th *= 2 / 3.141592; // # use 2/PI;
            th = (th > 1) ? 1 : th;

            // 0.05 - k - 0.8 on u
            float min = 0.05f;
            float max = 0.8f;

            Dx::Point k = u * ((1 - th) * (max - min) + min);
            k = k * TS;

            return k;
        }

        void deform(std::vector<float>* deformedCvsArr)
        {
            // deform rachis
            for( int i = 0; i < m_SourceFeather.m_vRachisUParams.size(); ++i)
            {
                float p = m_SourceFeather.m_vRachisUParams[i];

                // disp
                m_ncRachis.position(p);

                Dx::Point disp = m_ncRachis.m_vPos[0] - m_SourceFeather.m_vRachis_uInitCVs[i];

                Dx::Point cv = m_SourceFeather.m_vRachisCVs[i];
                cv = cv + disp;

                // rachis CVS set position
                if (i == 0 || i == m_SourceFeather.m_vRachisUParams.size() - 1)
                {
                    for(int cvs = 0; cvs < 3; cvs++)
                        deformedCvsArr->push_back(cv[cvs]);
                }
                for(int cvs = 0; cvs < 3; cvs++)
                    deformedCvsArr->push_back(cv[cvs]);
            }

            // deform barbs
            for(int s = 0; s < 2; ++s)
            {
                for (int b = 0; b < m_SourceFeather.m_vBarbUParams[s].size(); ++b)
                {
                    float rachisParam = m_SourceFeather.m_vBarbParams[s][b][0];
                    float edgeParam = m_SourceFeather.m_vBarbParams[s][b][1];

                    m_ncRachis.position(rachisParam, 1);
                    m_ncEdges[s].position(edgeParam);
                    Dx::Point r = m_ncRachis.m_vPos[0];
                    Dx::Point F = m_ncRachis.m_vPos[1];
                    Dx::Point e = m_ncEdges[s].m_vPos[0];
                    Dx::Point u = m_SourceFeather.m_vBarb_uVectors[s][b];

                    // get tangent space
                    Dx::Matrix TS = barbTangentSpace(rachisParam, r, F);

                    // r - k - e curve
                    Dx::Point k = getK(e, u, TS);
                    // std::cout << "===k===" << std::endl;
                    // k.Print();
                    Dx::NurbsCurve rke = Dx::NurbsCurve({r, k, e}, 2);

                    // ratio of k between r and e
                    float rkLength = (r - k).length();
                    float ekLength = (e - k).length();

                    float rat = rkLength / (rkLength + ekLength);

                    // deform barbs
                    for(int i = 0; i < m_SourceFeather.m_vBarbUParams[s][b].size(); ++i)
                    {
                        float p = m_SourceFeather.m_vBarbUParams[s][b][i];
                        p = reparameter(p, rat);

                        if ( p >= 0.99999f)
                        {
                            p = 0.99999f;
                        }
                        else if ( p <= 0.00001f)
                        {
                            p = 0.00001f;
                        }


                        rke.position(p);
                        Dx::Point disp = rke.m_vPos[0] - m_SourceFeather.m_vBarb_uInitCVs[s][b][i];

                        Dx::Point cv = m_SourceFeather.m_vBarbCVs[s][b][i];
                        cv = cv + disp;

                        // $1
                        if(isnan(cv.x) || isnan(cv.y) || isnan(cv.z))
                        {
                            std::cout << "------------------------- [Nan] : " << s << ", " << b << std::endl;
                            TS.Print();
                        }

                        // barbsCurve CVS Set Position
                        if (i == 0 || i == m_SourceFeather.m_vBarbUParams[s][b].size() - 1)
                        {
                            for(int cvs = 0; cvs < 3; cvs++)
                                deformedCvsArr->push_back(cv[cvs]);
                        }
                        for(int cvs = 0; cvs < 3; cvs++)
                            deformedCvsArr->push_back(cv[cvs]);
                    }
                }
            }
        }
    };
}
