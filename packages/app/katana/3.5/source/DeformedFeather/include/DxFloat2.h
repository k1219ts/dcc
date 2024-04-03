#pragma once

namespace Dx
{
    class Float2
    {
        // variarble
    public:
        float x;
        float y;

    public:
        Float2() // Default Construct
        {
            x = 0;
            y = 0;
        }
        Float2(const Float2& v)
        {
            x = v.x;
            y = v.y;
        }
        Float2(const float& _x, const float& _y)
        {
            x = _x;
            y = _y;
        }

    public:
        float& operator[](const int& i)
        {
            switch(i)
            {
                default:
                case 0: { return x; }
                case 1: { return y; }
            }
        }
    };
}
