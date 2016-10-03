#ifndef TOY_CNN_TESTS_PRINT_UTILS_HPP__
#define TOY_CNN_TESTS_PRINT_UTILS_HPP__

#include <ostream>

#include <utils/Matrix.hpp>

namespace test
{

std::ostream& operator << (std::ostream &inStream, const utils::Dimension &inDim);

template<typename TDataType>
std::ostream& operator << (std::ostream &inStream, const utils::Matrix<TDataType> &inMatrix);

}

#endif // TOY_CNN_TESTS_PRINT_UTILS_HPP__
