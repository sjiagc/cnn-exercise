#include "PrintUtils.hpp"

#include <utils/Dimension.hpp>

namespace test
{

std::ostream& operator << (std::ostream &inStream, const utils::Dimension &inDim)
{
    return inStream << inDim.getX() << ", "
              << inDim.getY() << ", "
              << inDim.getZ() << ", "
              << inDim.getW();
}

template<typename TDataType>
std::ostream& operator << (std::ostream &inStream, const utils::Matrix<TDataType> &inMatrix)
{
    const utils::Dimension &theDim = inMatrix.getDimension();
    const typename utils::Matrix<TDataType>::data_type *theData = inMatrix.getData();
    for (int64_t w = 0; w < theDim.getW(); ++w) {
        for (int64_t z = 0; z < theDim.getZ(); ++z) {
            inStream << "(" << w << ", " << z << "):" << std::endl;
            for (int64_t x = 0; x < theDim.getX(); ++x) {
                for (int64_t y = 0; y < theDim.getY(); ++y) {
                    inStream << theData[inMatrix.offset(x, y, z, w)] << ", ";
                }
                inStream << std::endl;
            }
        }
    }
    return inStream;
}

template std::ostream& operator << (std::ostream &inStream, const utils::Matrix<double> &inMatrix);

}
