#ifndef INFINI_C_OPS_H_
#define INFINI_C_OPS_H_

INFINI_OPS_API InfiniOpsStatus infiniOpsAdd(InfiniOpsHandle handle,
                                            InfiniOpsConfig config,
                                            const InfiniOpsTensor* input,
                                            const InfiniOpsTensor* other,
                                            InfiniOpsTensor* out);

#endif  // INFINI_C_OPS_H_
