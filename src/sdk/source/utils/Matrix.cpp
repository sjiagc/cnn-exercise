#include <utils/Matrix.hpp>

#include "DataBlock.hpp"

#include <stdexcept>

namespace utils
{

template<typename TDataType>
Matrix<TDataType>::Matrix(const Dimension &inDimension)
    : m_dimension(inDimension)
    , m_startOffset(0)
    , m_stride(1,
               m_dimension.getX(),
               m_dimension.getX() * m_dimension.getY(),
               m_dimension.getX() * m_dimension.getY() * m_dimension.getZ())
    , m_data(new DataBlock<TDataType>(m_dimension.getX() * m_dimension.getY() * m_dimension.getZ() * m_dimension.getW()))
{
}

template<typename TDataType>
Matrix<TDataType>::Matrix(const Matrix &inSrc)
    : m_dimension(inSrc.m_dimension)
    , m_startOffset(inSrc.m_startOffset)
    , m_stride(inSrc.m_stride)
    , m_data(inSrc.m_data)
{
}

template<typename TDataType>
Matrix<TDataType>::Matrix(Matrix &&inSrc)
    : m_dimension(inSrc.m_dimension)
    , m_startOffset(inSrc.m_startOffset)
    , m_stride(inSrc.m_stride)
    , m_data(inSrc.m_data)
{
    inSrc.m_data = nullptr;
}

template<typename TDataType>
Matrix<TDataType>::Matrix(const Matrix &inSrc, const Dimension &inStartDimension, const Dimension &inEndDimension)
    : Matrix(inSrc)
{
    if (inEndDimension <= inStartDimension ||
            inStartDimension >= m_dimension ||
            inEndDimension > m_dimension)
        throw std::runtime_error("Matrix::sub: invalid end and start dimension");
    size_t theNewOffset = m_stride.getW() * inStartDimension.getW() +
            m_stride.getZ() * inStartDimension.getZ() +
            m_stride.getY() * inStartDimension.getY() +
            m_stride.getX() * inStartDimension.getX();
    m_startOffset += theNewOffset;
    m_dimension = inEndDimension - inStartDimension;
    if (m_dimension.getW() == 0) m_dimension.setW(1);
    if (m_dimension.getZ() == 0) m_dimension.setZ(1);
    if (m_dimension.getY() == 0) m_dimension.setY(1);
    if (m_dimension.getX() == 0) m_dimension.setX(1);
}

template<typename TDataType>
Matrix<TDataType>
Matrix<TDataType>::sub(const Dimension &inStartDimension, const Dimension &inEndDimension) const
{
    return Matrix<TDataType>(*this, inStartDimension, inEndDimension);
}

template<typename TDataType>
Matrix<TDataType>::~Matrix()
{
}

template<typename TDataType>
Matrix<TDataType>&
Matrix<TDataType>::operator = (const Matrix<TDataType> &inArg)
{
    if (m_dimension != inArg.m_dimension)
        throw std::runtime_error("Matrix::operator=: dimension not match");
    //m_dimension = inArg.m_dimension;
    m_startOffset = inArg.m_startOffset;
    m_stride = inArg.m_stride;
    m_data = inArg.m_data;
    return *this;
}

template<typename TDataType>
const TDataType*
Matrix<TDataType>::getData() const
{
    return m_data->getData() + m_startOffset;
}

template<typename TDataType>
TDataType*
Matrix<TDataType>::getMutableData()
{
    return m_data->getMutableData() + m_startOffset;
}

template<typename TDataType>
const TDataType*
Matrix<TDataType>::getGPUData() const
{
    return m_data->getGPUData() + m_startOffset;
}

template<typename TDataType>
TDataType*
Matrix<TDataType>::getMutableGPUData()
{
    return m_data->getMutableGPUData() + m_startOffset;
}

template<typename TDataType>
const Dimension&
Matrix<TDataType>::getDimension() const
{
    return m_dimension;
}

template<typename TDataType>
const Dimension&
Matrix<TDataType>::getStride() const
{
    return m_stride;
}

template<typename TDataType>
bool
Matrix<TDataType>::reshape(const Dimension &inDimension)
{
    if (inDimension.count() == m_dimension.count()) {
        DataBlock<TDataType> *theNewData =
                new DataBlock<TDataType>(inDimension.getX() * inDimension.getY() * inDimension.getZ() * inDimension.getW());
        int64_t theNewDataIndex = 0;
        TDataType *theNewDataPtr = theNewData->getMutableData();
        const TDataType *theOriginDataPtr = m_data->getData();
        for (int64_t w = 0, theWCount = m_dimension.getW(); w < theWCount; ++w) {
            for (int64_t z = 0, theZCount = m_dimension.getZ(); z < theZCount; ++z) {
                for (int64_t y = 0, theYCount = m_dimension.getY(); y < theYCount; ++y) {
                    for (int64_t x = 0, theXCount = m_dimension.getX(); x < theXCount; ++x) {
                        theNewDataPtr[theNewDataIndex++] = theOriginDataPtr[offset(x, y, z, w)];
                    }
                }
            }
        }

        m_data.reset(theNewData);
        m_dimension  = inDimension;
        m_startOffset = 0;
        m_stride = Dimension(1,
                             m_dimension.getX(),
                             m_dimension.getX() * m_dimension.getY(),
                             m_dimension.getX() * m_dimension.getY() * m_dimension.getZ());
        return true;
    }
    return false;
}

template<typename TDataType>
int64_t
Matrix<TDataType>::offset(int64_t inX = 0, int64_t inY = 0, int64_t inZ = 0, int64_t inW = 0) const
{
    if (inX >= m_dimension.getX() || inX < 0 ||
            inY >= m_dimension.getY() || inY < 0 ||
            inZ >= m_dimension.getZ() || inZ < 0 ||
            inW >= m_dimension.getW() || inW < 0)
        throw std::runtime_error("Matrix::offset: invalid indices");
    return m_stride.getW() * inW +
            m_stride.getZ() * inZ +
            m_stride.getY() * inY +
            m_stride.getX() * inX;
}

template class Matrix<double>;

}
