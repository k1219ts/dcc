#pragma once

#include "DxPoint.h"
#include <vector>
#include <string>

namespace Dx
{
    class NurbsCurve
    {
    public:
        std::vector<Dx::Point>     m_vCvList;
        int                         m_nDegree;
        std::vector<int>            m_vKnotList;
        Dx::Point                  m_vPos[2];

    public:
        NurbsCurve(const std::vector<Dx::Point> cv, const int& degree = 3)
        {
            m_vCvList = cv;
            m_nDegree = degree;

            for(int i = 0; i < 2; ++i)
                m_vPos[i] = Dx::Point();

            getKnot();
        }
        NurbsCurve()
        {
            m_vCvList = std::vector<Dx::Point>();
            m_nDegree = 0;
            m_vKnotList = std::vector<int>();
            for(int i = 0; i < 2; ++i)
                m_vPos[i] = Dx::Point();
        }

    public:
        bool getKnot()
        {
            int knotCount = m_vCvList.size() - (m_nDegree - 1);

            if (knotCount == 0)
                return false;

            // initialize
            for(int i = 0; i < knotCount; ++i)
            {
                m_vKnotList.push_back(i);
            }

            //  insert 0, -1 index
            int lastKnot = m_vKnotList.back();
            for(int i = 0; i < (int)m_nDegree; ++i)
            {
                m_vKnotList.insert(m_vKnotList.begin(), 1, 0.0f);
                m_vKnotList.push_back(lastKnot);
            }

            return true;
        }

        int findSpan(const float& u)
        {
            int span = 0;
            int numCVs = m_vCvList.size() + 1;

            if (u < m_vKnotList.back())
            {
                while(span < numCVs && m_vKnotList[span] <= u)
                {
                    span += 1;
                }
            }
            else
            {
                while(span < numCVs && m_vKnotList[span] < u)
                {
                    span += 1;
                }
            }

            return span -1;
        }

        std::vector<std::vector<float>> derivative(const float& _u, const int& span, int tangent = 0)
        {
            int order = (int)tangent;
            float u = _u;

            // Initialize Variables
            float left[m_nDegree + 1];
            float right[m_nDegree + 1];
            float ndu[m_nDegree + 1][m_nDegree + 1];

            for(int i = 0; i < m_nDegree + 1; i++)
            {
                left[i] = 1.0f;
                right[i] = 1.0f;
                for(int j = 0; j < m_nDegree + 1; j++)
                    ndu[i][j] = 1.0f;
            }

            // // N[0][0] = 1.0 by definition
            for(int j = 1; j < m_nDegree + 1; ++j)
            {
                left[j] = u - m_vKnotList[span + 1 - j];
                right[j] = m_vKnotList[span + j] - u;
                float saved = 0.0f;

                for(int r = 0; r < j; r++)
                {
                    // Lower triangle
                    ndu[j][r] = right[r + 1] + left[j - r];
                    float temp = ndu[r][j - 1] / ndu[j][r];

                    // Upper triangle
                    ndu[r][j] = saved + (right[r + 1] * temp);
                    saved = left[j - r] * temp;
                }
                ndu[j][j] = saved;
            }

            // // Load the basis functions
            auto min = (m_nDegree > order) ? order + 1 : m_nDegree + 1;
            std::vector<std::vector<float>> ders;
            for(int i = 0; i < min; i++)
            {
                ders.push_back(std::vector<float>());
                for(int j = 0; j < (m_nDegree + 1); j++)
                {
                    ders[i].push_back(0.0f);
                }
            }

            for(int j = 0; j < m_nDegree + 1; ++j)
            {
                ders[0][j] = ndu[j][m_nDegree];
            }

            // Start calcuation derivatives
            float a[2][(m_nDegree + 1)];

            for(int i = 0; i < 2; i++)
            {
                for(int j = 0; j < m_nDegree + 1; j++)
                {
                    a[i][j] = 1.0f;
                }
            }

            // Loop over function index
            for(int r = 0; r < m_nDegree + 1; ++r)
            {
                // Alternate rows in array a
                int s1 = 0;
                int s2 = 1;
                a[0][0] = 1.0f;

                // Loop to compute k-th derivative
                for(int k = 1; k < order + 1; ++k)
                {
                    float d = 0.0f;
                    int rk = r - k;
                    int pk = m_nDegree - k;

                    if (r >= k)
                    {
                        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                        d = a[s2][0] * ndu[rk][pk];
                    }

                    int j1 = (rk >= -1) ? 1 : -rk;
                    int j2 = (r - 1 <= pk) ? k - 1 : m_nDegree - r;

                    for(int j = j1; j < j2 + 1; ++j)
                    {
                        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                        d += (a[s2][j] * ndu[rk + j][pk]);
                    }

                    if (r <= pk)
                    {
                        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                        d += (a[s2][k] * ndu[r][pk]);
                    }
                    ders[k][r] = d;

                    // Switch rows
                    int tmp = s1;
                    s1 = s2;
                    s2 = tmp;
                }
            }

            // Multiply through by the correct factors
            float r = (float)m_nDegree;
            for(int k = 1; k < order + 1; ++k)
            {
                for(int j = 0; j < m_nDegree + 1; ++j)
                {
                    ders[k][j] *= r;
                }
                r *= (m_nDegree - k);
            }

            return ders;
        }

        void position(const float& u, int tangent = 0)
        {
            int order                                       = tangent + 1;
            int span                                        = findSpan(u);
            std::vector<std::vector<float>> ders            = derivative(u, span, tangent);

            for(int i = 0; i < 2; ++i)
                m_vPos[i] = Dx::Point();

            for(int k = 0; k < order; ++k)
            {
                for(int j = 0; j < m_nDegree + 1; ++j)
                {
                    for(int i = 0; i < 3; i++)
                    {
                        float cv = m_vCvList[span - m_nDegree + j][i];
                        m_vPos[k][i] += ders[k][j] * cv;
                    }
                }
            }
        }
    };
}
