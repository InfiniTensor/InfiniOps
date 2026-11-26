#include <infini/ops.h>

#include "native/cuda/hygon/ops/per_tensor_dequant_i8/kernel.h"
#include "native/cuda/hygon/ops/per_tensor_quant_i8/kernel.h"

extern "C" int check_tensor_i8(void* floating, void* packed, float* scale,
                               float* zero, int dtype, int rank,
                               const size_t* shape,
                               const ptrdiff_t* float_strides,
                               const ptrdiff_t* packed_strides, int dequantize,
                               int is_static, void* stream) {
  using namespace infini::ops;
  try {
    const Tensor::Shape dims(shape, shape + rank);
    const Tensor::Strides fs(float_strides, float_strides + rank);
    const Tensor::Strides ps(packed_strides, packed_strides + rank);
    const Device device{Device::Type::kHygon, 0};
    const DataType type = dtype == 0   ? DataType::kFloat32
                          : dtype == 1 ? DataType::kFloat16
                                       : DataType::kBFloat16;
    const Tensor x(floating, dims, type, device, fs);
    const Tensor q(packed, dims, DataType::kInt8, device, ps);
    const Tensor s(scale, Tensor::Shape{1}, DataType::kFloat32, device);
    const auto z =
        zero ? std::optional<Tensor>{Tensor(zero, Tensor::Shape{1},
                                            DataType::kFloat32, device)}
             : std::nullopt;
    if (dequantize >= 2) {
      Handle handle;
      handle.set_stream(stream);
      if (dequantize & 1) {
        const auto config = PerTensorDequantI8::DefaultConfig(device.type());
        PerTensorDequantI8::Call(handle, config, x, q, s, z);
      } else {
        const auto config = PerTensorQuantI8::DefaultConfig(device.type());
        const bool fixed = is_static;
        PerTensorQuantI8::Call(handle, config, x, q, s, z, fixed);
      }
    } else if (dequantize) {
      Operator<PerTensorDequantI8, Device::Type::kHygon> op(x, q, s, z);
      op.set_stream(stream);
      op(x, q, s, z);
    } else {
      Operator<PerTensorQuantI8, Device::Type::kHygon> op(x, q, s, z,
                                                          bool(is_static));
      op.set_stream(stream);
      op(x, q, s, z, bool(is_static));
    }
    return 0;
  } catch (const std::invalid_argument&) {
    return 1;
  } catch (...) {
    return 2;
  }
}
