#include "jit.h"

#include <cuda.h>

#include <functional>
#include <string>

#include "cache.h"

namespace infini::ops {

DeviceInfo CurrentDevice() {
  DeviceInfo info;
  CUdevice dev;
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) return info;
  info.id = static_cast<int>(dev);
  int major = 0, minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       dev);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       dev);
  info.arch = major * 10 + minor;
  return info;
}

static CUresult LoadCubin(const char* cubin_path, const char* meta_path,
                          CUfunction* out_func, unsigned* out_shared,
                          CUmodule* out_mod) {
  if (!FileExists(meta_path)) return CUDA_ERROR_FILE_NOT_FOUND;
  std::string meta_json = ReadFile(meta_path);
  int shared = JsonGetInt(meta_json, "shared", 0);
  if (shared < 0) shared = 0;
  std::string fn_name = JsonGetString(meta_json, "name", "kernel");

  int global_scratch = JsonGetInt(meta_json, "global_scratch_size", 0);
  int profile_scratch = JsonGetInt(meta_json, "profile_scratch_size", 0);
  if (global_scratch > 0 || profile_scratch > 0) {
    fprintf(stderr, "triton jit: scratch not supported yet\n");
    return CUDA_ERROR_NOT_SUPPORTED;
  }

  CUmodule mod;
  CUresult err = cuModuleLoad(&mod, cubin_path);
  if (err != CUDA_SUCCESS) return err;
  CUfunction func;
  err = cuModuleGetFunction(&func, mod, fn_name.c_str());
  if (err != CUDA_SUCCESS) {
    cuModuleUnload(mod);
    return err;
  }

  if (shared > 49152) {
    CUdevice dev;
    err = cuCtxGetDevice(&dev);
    if (err != CUDA_SUCCESS) {
      cuModuleUnload(mod);
      return err;
    }
    int optin = 0;
    cuDeviceGetAttribute(
        &optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
    int st = 0;
    cuFuncGetAttribute(&st, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
    if (shared > optin - st) {
      cuModuleUnload(mod);
      return CUDA_ERROR_INVALID_VALUE;
    }
    cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED);
    err = cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin - st);
    if (err != CUDA_SUCCESS) {
      cuModuleUnload(mod);
      return err;
    }
  }

  *out_func = func;
  *out_shared = static_cast<unsigned>(shared);
  *out_mod = mod;
  return CUDA_SUCCESS;
}

void* GetKernel(const char* op_name, const char* signature_str, void* stream,
                const TritonConfig& opts, unsigned* out_shared) {
  DeviceInfo dev = CurrentDevice();

  auto r = CacheQuery(op_name, signature_str, opts.num_warps, opts.num_stages,
                      dev.arch, dev.id);
  if (r.mem_hit) {
    *out_shared = r.shared;
    return r.func;
  }

  std::string cubin_path = r.out_prefix + ".cubin";
  std::string meta_path = r.out_prefix + ".json";

  if (!CacheComplete(cubin_path, meta_path)) {
    int ret = CompileKernel(op_name, r.out_prefix.c_str(), opts.num_warps,
                            opts.num_stages, dev.id, signature_str);
    if (ret != 0) return nullptr;
  }

  CUfunction func;
  unsigned shared;
  CUmodule mod;
  CUresult err =
      LoadCubin(cubin_path.c_str(), meta_path.c_str(), &func, &shared, &mod);
  if (err != CUDA_SUCCESS) return nullptr;

  KernelCacheEntry mine{static_cast<void*>(func), shared};
  KernelCacheEntry winner;
  if (KernelCacheLookup(r.mem_key, &winner)) {
    cuModuleUnload(mod);
    func = static_cast<CUfunction>(winner.func);
    shared = winner.shared;
  } else {
    KernelCacheInsert(r.mem_key, mine);
  }

  *out_shared = shared;
  return static_cast<void*>(func);
}

int LaunchKernel(const char* op_name, const char* signature_str, void* stream,
                 Grid grid, TritonConfig opts, void** args) {
  CUstream cu_stream = static_cast<CUstream>(stream);
  unsigned shared = 0;
  void* func_ptr = GetKernel(op_name, signature_str, stream, opts, &shared);
  if (func_ptr == nullptr) return static_cast<int>(CUDA_ERROR_UNKNOWN);

  return static_cast<int>(cuLaunchKernel(
      static_cast<CUfunction>(func_ptr), grid.x, grid.y, grid.z,
      opts.num_warps * 32, 1, 1, shared, cu_stream, args, nullptr));
}

}  // namespace infini::ops
