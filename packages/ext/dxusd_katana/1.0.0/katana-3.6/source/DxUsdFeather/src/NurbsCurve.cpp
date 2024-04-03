#include <OpenEXR/ImathVec.h>
#include <OpenEXR/ImathMatrix.h>

#include "NurbsCurve.h"
#include "Utils.h"

namespace DxUsdFeather
{


void NurbsCurve::Print(bool debug)
{
    std::cout << SHAP << "\n# NurbsCurve (d = " << degree << ")";
    std::cout << std::endl;
    std::cout << DASH << "\n# - Knots" << std::endl;
    for(auto k : knot)
        std::cout << k << ", ";
    std::cout << std::endl;
    std::cout << DASH << "\n# - CVs" << std::endl;
    PrintVecs(CV, "# ");
}

int NurbsCurve::GetSpan(float u)
{
    int span = 0;
    if(u < knot.back())
    {
        while(span <= CV.size() && knot[span] <= u)
            span += 1;
    }
    else
    {
        while(span <= CV.size() && knot[span] < u)
            span += 1;
    }
    return span - 1;
}

std::vector<std::vector<float>> NurbsCurve::Derivative(
    float u, int span, int order
)
{
    // Initialize Variables
    float left[degree + 1];
    float right[degree + 1];
    float ndu[degree + 1][degree + 1];

    for(int i = 0; i <= degree; i++)
    {
        left[i] = 1.0f;
        right[i] = 1.0f;
        for(int j = 0; j <= degree; j++)
            ndu[i][j] = 1.0f;
    }

    // N[0][0] = 1.0 by definition
    for(int i = 1; i <= degree; i++)
    {
        left[i]  = u - knot[span + 1 - i];
        right[i] = knot[span + i] - u;
        float saved = 0.0f;

        for(int j = 0; j < i; j++)
        {
            // Lower triangle
            ndu[i][j] = right[j + 1] + left[i - j];
            float temp = ndu[j][i - 1] / ndu[i][j];

            // Upper triangle
            ndu[j][i] = saved + (right[j + 1] * temp);
            saved = left[i - j] * temp;
        }
        ndu[i][i] = saved;
    }

    // Load the basis functions
    auto min = (degree > order-1) ? order: degree + 1;
    std::vector<std::vector<float>> ders;
    for(int i = 0; i < min; i++)
    {
        ders.push_back(std::vector<float>());
        for(int j = 0; j <= degree; j++)
            ders[i].push_back(0.0f);
    }

    for(int i = 0; i <= degree; i++)
        ders[0][i] = ndu[i][degree];

    // Start calcuation derivatives
    float a[2][degree + 1];
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j <= degree; j++)
            a[i][j] = 1.0f;
    }

    // Loop over function index
    for(int i = 0; i <= degree; i++)
    {
        // Alternate rows in array a
        int s1 = 0; int s2 = 1;
        a[0][0] = 1.0f;

        // Loop to compute k-th derivative
        for(int j = 1; j < order; j++)
        {
            float d = 0.0f;
            int rk = i - j;
            int pk = degree - j;

            if (i >= j)
            {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }

            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (i - 1 <= pk) ? j - 1 : degree - i;

            for(int k = j1; k <= j2; ++k)
            {
                a[s2][k] = (a[s1][k] - a[s1][k - 1]) / ndu[pk + 1][rk + k];
                d += (a[s2][k] * ndu[rk + k][pk]);
            }

            if (i <= pk)
            {
                a[s2][j] = -a[s1][j - 1] / ndu[pk + 1][i];
                d += (a[s2][j] * ndu[i][pk]);
            }
            ders[j][i] = d;

            // Switch rows
            int tmp = s1;
            s1 = s2;
            s2 = tmp;
        }
    }

    // Multiply through by the correct factors
    float r = (float)degree;
    for(int i = 1; i < order; i++)
    {
        for(int j = 0; j <= degree; j++)
            ders[i][j] *= r;

        r *= degree - i;
    }

    return ders;
}

Imath::V3f NurbsCurve::GetPosition(float u, bool tangent)
{
    int order = tangent ? 2 : 1;
    u *= knot.back();

    if(u < 0.0f)
        u = 0.0f;
    if(u > knot.back())
        u = (float)knot.back();

    Imath::V3f pos[order];
    // init pos
    for(int i = 0; i < order; i++)
        pos[i] = Imath::V3f(0.f, 0.f, 0.f);

    int span = GetSpan(u);
    std::vector<std::vector<float>> ders = Derivative(u, span, order);

    // std::cout << "span : " << span << std::endl;
    // for(auto der : ders)
    // {
    //     std::cout << "ders ----------------" << std::endl;
    //     for(auto d : der)
    //         std::cout << d << ", ";
    //     std::cout << std::endl;
    // }
    // std::cout << DASH << std::endl;

    for(int k = 0; k < order; k++)
    {
        for(int j = 0; j <= degree; j++)
        {
            for(int i = 0; i < 3; i++)
            {
                float cv = CV[span - degree + j][i];
                pos[k][i] += ders[k][j] * cv;
                // std::cout << k << ", " << j << ", " << i << " : ";
                // std::cout << cv << " ## " << pos[k][i] << std::endl;
            }
        }
    }
    return pos[order-1];
}



} //DxUsdFeather
