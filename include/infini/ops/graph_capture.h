#ifndef INFINI_OPS_GRAPH_CAPTURE_H_
#define INFINI_OPS_GRAPH_CAPTURE_H_

#include <infini/rt.h>

#include <memory>

namespace infini::ops {

// Owns provider allocations referenced by a captured device graph. Create this
// before the runtime begins capture, call EndCapture after capture has ended
// (including an aborted capture), and retain it until the graph and every
// executable referencing its allocations have been destroyed.
class GraphCaptureMemory {
 public:
  virtual ~GraphCaptureMemory() = default;
  virtual void EndCapture() = 0;
};

// Returns nullptr for providers that do not need a separate capture pool.
// The stream must remain alive until EndCapture. Capture and EndCapture must
// run on the same host thread. Destruction also ends an unfinished scope.
std::shared_ptr<GraphCaptureMemory> BeginGraphCaptureMemory(
    const infini::rt::Device& device, void* stream);

}  // namespace infini::ops

#endif  // INFINI_OPS_GRAPH_CAPTURE_H_
