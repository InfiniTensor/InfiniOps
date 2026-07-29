#include <pybind11/embed.h>

#include <cstdio>
#include <mutex>

#include "cache.h"
#include "jit.h"

namespace infini::ops {

bool compiler_init() {
  static std::once_flag flag;
  static bool ready = false;

  std::call_once(flag, [] {
    namespace py = pybind11;

    auto setup = [] { py::module_::import("infini.triton.jit.compile"); };

    if (Py_IsInitialized()) {
      py::gil_scoped_acquire gil;
      try {
        setup();
        ready = true;
      } catch (const py::error_already_set& e) {
        fprintf(stderr, "jit init: %s\n", e.what());
      }
    } else {
      py::initialize_interpreter(false);
      try {
        setup();
        ready = true;
      } catch (const py::error_already_set& e) {
        fprintf(stderr, "jit init: %s\n", e.what());
      }
      (void)PyEval_SaveThread();
    }
  });

  return ready;
}

int compile_kernel(const char* op_name, const char* out_prefix, int num_warps,
                   int num_stages, int device_id, const char* signature) {
  if (!compiler_init()) return -1;

  namespace py = pybind11;
  py::gil_scoped_acquire gil;
  try {
    py::module_ mod = py::module_::import("infini.triton.jit.compile");
    mod.attr("_do_compile")(op_name, out_prefix, num_warps, num_stages,
                            device_id, signature);
    return 0;
  } catch (const py::error_already_set& e) {
    fprintf(stderr, "jit compile: %s\n", e.what());
    return -2;
  }
}

config_t autotune_bench(const char* op_name,
                        const std::vector<config_t>& configs,
                        const std::string& sig, const std::vector<void*>& ptrs,
                        const std::vector<grid_t>& grids, int warmup, int rep,
                        const char* key, int device_id) {
  config_t cached;
  if (autotune_cache_lookup(key, &cached)) return cached;

  namespace py = pybind11;
  if (!compiler_init()) return configs.empty() ? config_t{} : configs[0];
  py::gil_scoped_acquire gil;
  try {
    py::module_ mod = py::module_::import("infini.triton.jit.compile");

    device_info_t dev = current_device();

    py::list cands;
    for (const auto& c : configs) {
      py::dict cd;
      cd["num_warps"] = c.num_warps;
      cd["num_stages"] = c.num_stages;
      py::list ce;
      for (const auto& [k, v] : c.constexprs) {
        py::tuple kv(2);
        kv[0] = k;
        kv[1] = v;
        ce.append(kv);
      }
      cd["constexprs"] = ce;

      std::string full_sig = sig;
      for (const auto& [k, v] : c.constexprs)
        full_sig += k + "=" + std::to_string(v) + ",";
      if (!full_sig.empty() && full_sig.back() == ',') full_sig.pop_back();
      cd["full_sig"] = full_sig;
      cd["out_prefix"] = std::string(TRITON_JIT_CACHE_DIR) + "/" +
                         cache_file_key(op_name, full_sig.c_str(), c.num_warps,
                                        c.num_stages, dev.arch);

      cands.append(cd);
    }

    py::list args;
    size_t ptr_idx = 0;
    size_t pos = 0;
    while (pos < sig.size()) {
      size_t comma = sig.find(',', pos);
      std::string part = sig.substr(pos, comma - pos);
      pos = (comma == std::string::npos) ? sig.size() : comma + 1;
      if (part.empty()) continue;

      if (part[0] == '*') {
        uint64_t val = *static_cast<uint64_t*>(ptrs[ptr_idx++]);
        args.append(static_cast<int64_t>(val));
      } else if (part.find(":1") != std::string::npos) {
        args.append(1);
      } else {
        uint64_t val = *static_cast<uint64_t*>(ptrs[ptr_idx++]);
        if (part.compare(0, 4, "fp32") == 0 || part.compare(0, 3, "f32") == 0) {
          args.append(*reinterpret_cast<float*>(&val));
        } else if (part.compare(0, 4, "fp64") == 0) {
          args.append(*reinterpret_cast<double*>(&val));
        } else {
          args.append(static_cast<int64_t>(val));
        }
      }
    }

    py::list grids_list;
    for (const auto& g : grids) {
      py::tuple t(3);
      t[0] = g.x;
      t[1] = g.y;
      t[2] = g.z;
      grids_list.append(t);
    }

    int best_idx = mod.attr("_do_autotune")(op_name, cands, args, grids_list,
                                            warmup, rep, device_id)
                       .cast<int>();
    if (best_idx < 0 || best_idx >= static_cast<int>(configs.size()))
      best_idx = 0;
    config_t winner = configs[best_idx];
    autotune_cache_insert(key, winner);
    return winner;
  } catch (const py::error_already_set& e) {
    fprintf(stderr, "jit autotune: %s\n", e.what());
    return configs.empty() ? config_t{} : configs[0];
  }
}

}  // namespace infini::ops
