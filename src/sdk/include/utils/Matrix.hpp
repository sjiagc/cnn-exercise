#ifndef TOY_CNN_SDK_UTILS_MATRIX_HPP__
#define TOY_CNN_SDK_UTILS_MATRIX_HPP__

#include "Dimension.hpp"

#include <memory>

namespace utils
{

template<typename TDataType> class DataBlock;

template<typename TDataType>
class Matrix
{
public:
    using data_type = TDataType;
    Matrix(const Dimension &inDimension);
    Matrix(const Matrix &inSrc);
    Matrix(Matrix &&inSrc);
    ~Matrix();

    Matrix& operator = (const Matrix &inArg);

    Matrix sub(const Dimension &inStartDimension, const Dimension &inEndDimension) const;

    const Dimension& getDimension() const;
    const Dimension& getStride() const;

    const TDataType* getData() const;
    TDataType* getMutableData();
    const TDataType* getGPUData() const;
    TDataType* getMutableGPUData();

    bool reshape(const Dimension &inDimension);

    int64_t offset(int64_t inX = 0, int64_t inY = 0, int64_t inZ = 0, int64_t inW = 0) const;

private:
    Matrix(const Matrix &inSrc, const Dimension &inStartDimension, const Dimension &inEndDimension);
    Dimension m_dimension;
    size_t m_startOffset;
    Dimension m_stride;
    std::shared_ptr<DataBlock<TDataType>> m_data;
};

}

#endif // TOY_CNN_SDK_UTILS_MATRIX_HPP__
