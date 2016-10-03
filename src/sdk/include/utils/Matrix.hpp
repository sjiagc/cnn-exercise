#ifndef TOY_CNN_SDK_UTILS_MATRIX_HPP__
#define TOY_CNN_SDK_UTILS_MATRIX_HPP__

#include "Types.hpp"

#include <memory>

namespace utils
{

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

    const TDataType* getData() const;
    TDataType* getData();
    const Dimension& getDimension() const;
    const Dimension& getStride() const;
    bool reshape(const Dimension &inDimension);

    int64_t offset(int64_t inX = 0, int64_t inY = 0, int64_t inZ = 0, int64_t inW = 0) const;

private:
    Matrix(const Matrix &inSrc, const Dimension &inStartDimension, const Dimension &inEndDimension);
    Dimension m_dimension;
    Dimension m_startDimension;
    Dimension m_stride;
    std::shared_ptr<TDataType> m_data;
};

}

#endif // TOY_CNN_SDK_UTILS_MATRIX_HPP__
