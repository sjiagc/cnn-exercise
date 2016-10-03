#ifndef TOY_CNN_SDK_LAYER_HPP__
#define TOY_CNN_SDK_LAYER_HPP__

#include "utils/Matrix.hpp"

#include <map>
#include <memory>
#include <string>

namespace layer
{

template<typename TDataType>
class Layer
{
public:
    using TLayerConfig = std::map<std::string, std::string>;
    using TUniqueHandle = std::unique_ptr<Layer<TDataType>, void(*)(Layer<TDataType>*)>;
    using TDataRestoring = std::map<std::string, const utils::Matrix<TDataType>*>;

    static TUniqueHandle create(const std::string &inLayerType, const TLayerConfig &inLayerConfig);

    virtual const char* getType() = 0;
    virtual void connect(Layer &inDescendentLayer) = 0;
    virtual void restore(const TDataRestoring &inStoredData) = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual const utils::Matrix<TDataType>* getOutput() const = 0;
    virtual const utils::Matrix<TDataType>* getDiff() const = 0;

    virtual void setForwardInput(const utils::Matrix<TDataType> &inInput) = 0;
    virtual void setBackwardDiff(const utils::Matrix<TDataType> &inDiff) = 0;
protected:
    virtual ~Layer() {}
private:
    static void release(Layer *inInstance);
};

}

#endif // TOY_CNN_SDK_LAYER_HPP__
