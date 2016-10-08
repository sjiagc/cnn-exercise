#ifndef TOY_CNN_SDK_TYPES_H__
#define TOY_CNN_SDK_TYPES_H__

#include <cstdint>

typedef short float16_t;
typedef float float32_t;
typedef double float64_t;

struct Dim
{
    int64_t x;
    int64_t y;
    int64_t z;
    int64_t w;
};

#endif // TOY_CNN_SDK_TYPES_H__
