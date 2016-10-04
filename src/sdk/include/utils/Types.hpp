#ifndef TOY_CNN_SDK_UTILS_TYPES_HPP__
#define TOY_CNN_SDK_UTILS_TYPES_HPP__

#include <cstdint>

using float16_t = short;
using float32_t = float;
using float64_t = double;

namespace utils
{

class Dimension
{
public:
    Dimension(int64_t inX, int64_t inY, int64_t inZ, int64_t inW)
    {
        m_dim[0] = inX;
        m_dim[1] = inY;
        m_dim[2] = inZ;
        m_dim[3] = inW;
    }
    Dimension(int64_t inX, int64_t inY, int64_t inZ)
        : Dimension(inX, inY, inZ, inZ > 0 ? 1 : 0)
    {}
    Dimension(int64_t inX, int64_t inY)
        : Dimension(inX, inY, inY > 0 ? 1 : 0)
    {}
    Dimension(int64_t inX)
        : Dimension(inX, inX > 0 ? 1 : 0)
    {}

    int64_t axisCount() const
    {
        for (int64_t theCount = 4; theCount > 0; --theCount) {
            if (m_dim[theCount - 1] > 1)
                return theCount;
        }
        return 1;
    }

    Dimension operator + (const Dimension &inArg) const
    {
        return Dimension(getX() + inArg.getX(), getY() + inArg.getY(), getZ() + inArg.getZ(), getW() + inArg.getW());
    }

    Dimension operator - (const Dimension &inArg) const
    {
        return Dimension(getX() - inArg.getX(), getY() - inArg.getY(), getZ() - inArg.getZ(), getW() - inArg.getW());
    }

    bool operator == (const Dimension &inArg) const
    {
        return getX() == inArg.getX() &&
                getY() == inArg.getY() &&
                getZ() == inArg.getZ() &&
                getW() == inArg.getW();
    }

    bool operator != (const Dimension &inArg) const
    {
        return !(*this == inArg);
    }

    bool operator <= (const Dimension &inArg) const
    {
        return getX() <= inArg.getX() &&
                getY() <= inArg.getY() &&
                getZ() <= inArg.getZ() &&
                getW() <= inArg.getW();
    }

    bool operator >= (const Dimension &inArg) const
    {
        return inArg <= *this;
    }

    bool operator < (const Dimension &inArg) const
    {
        return *this <= inArg && *this != inArg;
    }

    bool operator > (const Dimension &inArg) const
    {
        return inArg < *this;
    }

    int64_t getX() const { return m_dim[0]; }
    int64_t getY() const { return m_dim[1]; }
    int64_t getZ() const { return m_dim[2]; }
    int64_t getW() const { return m_dim[3]; }
    void setX(int64_t inX) { m_dim[0] = inX; }
    void setY(int64_t inY) { m_dim[1] = inY; }
    void setZ(int64_t inZ) { m_dim[2] = inZ; }
    void setW(int64_t inW) { m_dim[3] = inW; }

    int64_t count() const { return getX() * getY() * getZ() * getW(); }

private:
    int64_t m_dim[4];
};

}

#endif // TOY_CNN_SDK_UTILS_TYPES_HPP__
