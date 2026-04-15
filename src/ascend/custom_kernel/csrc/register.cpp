// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("rms_norm(Tensor input, Tensor weight, float eps=1e-6) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(ascend_kernel::rms_norm));
}
}  // namespace
