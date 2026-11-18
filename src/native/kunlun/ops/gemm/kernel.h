#ifndef INFINI_OPS_KUNLUN_GEMM_KERNEL_H_
#define INFINI_OPS_KUNLUN_GEMM_KERNEL_H_

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "native/cuda/ops/gemm/blas.h"  // BlasGemm: backend-parameterized
#include "native/kunlun/blas.h"
#include "native/kunlun/blas_utils.h"
#include "native/kunlun/elementwise_common.h"    // kunlun_launch::LaunchCast
#include "native/kunlun/launch2_declarations.h"  // LaunchFill
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun GEMM via xBLAS.
//
// xBLAS on the P800 supports f16 and f32 GEMMs natively, and mixed
// AB=bf16 + C=f32 GEMMs, but NOT all-bf16 GEMMs (status 15,
// CUBLAS_STATUS_NOT_SUPPORTED) — same limitation the legacy InfiniCore
// stack worked around. For bf16 outputs we therefore run the BLAS call
// with an f32 temporary output and cast f32 -> bf16 back into y.
template <>
class Operator<Gemm, Device::Type::kKunlun>
    : public BlasGemm<Blas<Device::Type::kKunlun>> {
 public:
  using BlasGemm<Blas<Device::Type::kKunlun>>::BlasGemm;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    (void)alpha;
    (void)beta;
    (void)trans_a;
    (void)trans_b;
    if (!y.IsContiguous()) {
      throw std::runtime_error("kunlun gemm: output must be contiguous");
    }
    if (a.strides().size() != 2 || b.strides().size() != 2) {
      // Batched calls keep 3-d shapes; fold leading batch into strides.
      if (a.strides().size() != 3 || b.strides().size() != 3 ||
          y.strides().size() != 3) {
        throw std::runtime_error("kunlun gemm: unsupported rank (2D/3D only)");
      }
    }
    if (a_type_ != b_type_ || y_type_ != a_type_) {
      throw std::runtime_error(
          "kunlun gemm: mixed dtypes unsupported (except bf16->f32 C)");
    }

    using Backend = Runtime<Device::Type::kKunlun>;
    if (y_type_ != DataType::kBFloat16) {
      // f16/f32: native BLAS path straight into y, then y += beta * c
      // (only beta == 1 with bias c; other betas throw — the fused op is
      //  not worth a second kernel for the 0/1 cases the suite exercises).
      RunBlasCore(a, b, y.data(), /*c_is_f32=*/false);
      if (beta_ != 0.0f && c.has_value()) {
        if (beta_ != 1.0f) {
          throw std::runtime_error(
              "kunlun gemm: f16/f32 supports beta only as 0/1 with bias c");
        }
        const unsigned long long elems = (unsigned long long)m_ *
                                         (unsigned long long)n_ *
                                         (unsigned long long)batch_count_;
        RunFlatAdd(c->data(), y.data(), elems, y_type_);
      }
      return;
    }

    // bf16: zeroed f32 workspace (up-cast bias c when beta_ != 0), GEMM with
    // C=f32, then cast f32 -> bf16 into y.
    const unsigned long long out_elems = (unsigned long long)m_ *
                                         (unsigned long long)n_ *
                                         (unsigned long long)batch_count_;
    void* y_f32 = nullptr;
    if (Backend::Malloc(&y_f32, out_elems * sizeof(float)) !=
        Backend::kSuccess) {
      throw std::runtime_error("kunlun gemm: f32 workspace alloc failed");
    }
    struct FreeGuard {
      void* p;
      ~FreeGuard() {
        if (p != nullptr) Runtime<Device::Type::kKunlun>::Free(p);
      }
    } free_guard{y_f32};
    if (beta_ != 0.0f && c.has_value()) {
      // C = beta * c: up-cast c to the f32 workspace, scaling if beta != 1.
      if (!c->IsContiguous() || (unsigned long long)c->numel() != out_elems) {
        throw std::runtime_error(
            "kunlun gemm: bf16 bias must be contiguous and match y");
      }
      kunlun_launch::LaunchCast(/*bf16 in*/ 2, /*f32 out*/ 0, y_f32, c->data(),
                                out_elems, stream_ ? stream_ : nullptr);
      if (beta_ != 1.0f) {
        kunlun_launch::LaunchScaleF32(y_f32, out_elems, beta_,
                                      stream_ ? stream_ : nullptr);
      }
    } else {
      long long shape[2] = {(long long)out_elems, 1};
      long long strides[2] = {1, 1};
      kunlun_launch::LaunchFill(4, 0, y_f32, 1, shape, strides, out_elems, 0.0f,
                                stream_ ? stream_ : nullptr);
    }
    RunBlasCore(a, b, y_f32, /*c_is_f32=*/true);
    kunlun_launch::LaunchCast(/*f32 in*/ 0, /*bf16 out*/ 2, y.data(), y_f32,
                              out_elems, stream_ ? stream_ : nullptr);
    {
      static const bool kDump = getenv("INFINI_KL_DUMP") != nullptr;
      static int yseq = 0;
      if (kDump && (yseq < 3 || n_ == 152064)) {
        int id = yseq++;
        std::string pf = "/tmp/kl_dump_" + std::to_string(id) + "_yf32.bin";
        std::string py = "/tmp/kl_dump_" + std::to_string(id) + "_ybf.bin";
        std::vector<char> buf(out_elems * 4), buf2(out_elems * 2);
        xpu_wait(static_cast<XPUStream>(stream_));
        xpu_memcpy_async(buf.data(), y_f32, buf.size(), XPU_DEVICE_TO_HOST,
                         static_cast<XPUStream>(stream_));
        xpu_memcpy_async(buf2.data(), y.data(), buf2.size(), XPU_DEVICE_TO_HOST,
                         static_cast<XPUStream>(stream_));
        xpu_wait(static_cast<XPUStream>(stream_));
        FILE* f = fopen(pf.c_str(), "wb");
        fwrite(buf.data(), 1, buf.size(), f);
        fclose(f);
        FILE* g = fopen(py.c_str(), "wb");
        fwrite(buf2.data(), 1, buf2.size(), g);
        fclose(g);
      }
    }
  }

 private:
  // Flat y += c over  (contiguous, same shape). Uploads fresh 1-d
  // metadata each call — negligible next to the GEMM itself.
  void RunFlatAdd(const void* c, void* y, unsigned long long elems,
                  DataType dtype) const {
    using Backend = Runtime<Device::Type::kKunlun>;
    const int tag = detail::KunlunDtypeTag(dtype);
    const std::size_t shape_bytes = sizeof(Tensor::Size);
    const std::size_t stride_bytes = sizeof(Tensor::Stride);
    const std::size_t meta_bytes = 3 * (shape_bytes + stride_bytes);
    const Tensor::Size h_shape = static_cast<Tensor::Size>(elems);
    const Tensor::Stride h_stride = 1;
    std::vector<std::byte> host(meta_bytes);
    std::byte* d_meta = nullptr;
    if (Backend::Malloc(reinterpret_cast<void**>(&d_meta), meta_bytes) !=
        Backend::kSuccess) {
      throw std::runtime_error("kunlun gemm: flat-add meta alloc failed");
    }
    struct FreeGuard {
      void* p;
      ~FreeGuard() {
        if (p != nullptr) Runtime<Device::Type::kKunlun>::Free(p);
      }
    } free_guard{d_meta};
    std::size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
      std::memcpy(host.data() + offset, &h_shape, shape_bytes);
      offset += shape_bytes;
      std::memcpy(host.data() + offset, &h_stride, stride_bytes);
      offset += stride_bytes;
    }
    if (Backend::Memcpy(d_meta, host.data(), meta_bytes,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess) {
      throw std::runtime_error("kunlun gemm: flat-add meta copy failed");
    }
    kunlun_launch::LaunchAdd(
        tag, y, y, c, reinterpret_cast<const unsigned long long*>(d_meta),
        reinterpret_cast<const unsigned long long*>(d_meta + shape_bytes +
                                                    stride_bytes),
        reinterpret_cast<const unsigned long long*>(
            d_meta + 2 * (shape_bytes + stride_bytes)),
        reinterpret_cast<const long long*>(d_meta + shape_bytes),
        reinterpret_cast<const long long*>(d_meta + shape_bytes * 2 +
                                           stride_bytes),
        reinterpret_cast<const long long*>(d_meta + shape_bytes * 3 +
                                           stride_bytes * 2),
        elems, 1, true, true, true, 1.0, stream_ ? stream_ : nullptr);
  }

  void RunBlasBf16(const Tensor& a, const Tensor& b, void* y_f32) const {
    RunBlasCore(a, b, y_f32, /*c_is_f32=*/true);
  }

  // BLAS call for the InfiniOps Gemm contract y[M,N] = a[M,K] @ b[K,N]
  // (alpha/beta handled by the caller). Column-major view: C[N,M] with
  // ldc = y.stride(-1). With B = a's memory (row-major [M,K] = col-major
  // [K,M], ldb = a.stride(-2)) the product C = op_a(A) @ op_b(B) needs
  // op_a(A) = [N,K] from b's memory:
  //   b row-major [K,N] (stride(-1)==1): memory IS col-major [N,K] ->
  //     op_a = N, lda = b.stride(-2).
  //   b permute view [K,N] with stride(-2)==1 (InfiniLM linear weights,
  //     memory = row-major weight [N,K] = col-major [K,N]): op_a = T turns
  //     it into [N,K], lda = b.stride(-1) = K.
  // B side: a row-major (stride(-1)==1) gives col-major [K,M] with op_b=N;
  // a col-major (stride(-2)==1) gives op_b=T, ldb = a.stride(-1).
  // When c_is_f32, C dtype is R_32F (bf16 A/B path); otherwise it matches
  // the compute dtype (f16/f32 native paths).
  void RunBlasCore(const Tensor& a, const Tensor& b, void* y_out,
                   bool c_is_f32) const {
    using BL = Blas<Device::Type::kKunlun>;
    const auto handle = this->GetHandle();
    if (BL::BlasSetStream(handle, static_cast<typename BL::Stream>(stream_)) !=
        0) {
      throw std::runtime_error("kunlun gemm: setting stream failed");
    }

    // BLAS A carries the [n, k] matrix of b (y = a @ b^T semantics).
    // xBLAS rejects op=N on an [m_arg, k] A whose leading dim is smaller
    // than m_arg (the permute-view case, probed standalone), so A is always
    // expressed as the col-major [k, n] of b's memory with op=T:
    //   b row-major [n, k] (contiguous): ld = stride(-2) = k.
    //   b permute view [k, n] (stride(-2)==1, InfiniLM linear): the memory
    //     as col-major [k, n] has ld = stride(-1) = k. op=T then computes
    //     A^T = the [n, k] matrix read at stride(-1) — verified equal to
    //     a @ b on the P800.
    const bool b_row_major = b.stride(-1) == 1;
    auto op_a = b_row_major ? BL::BLAS_OP_N : BL::BLAS_OP_T;
    const int lda_b = b_row_major ? (int)b.stride(-2) : (int)b.stride(-1);
    // a.stride(-1)==1 means row-major [M,K] = col-major [K,M]: op N.
    // Otherwise a is col-major [M,K] already: op T.
    auto op_b = a.stride(-1) == 1 ? BL::BLAS_OP_N : BL::BLAS_OP_T;

    auto dt_a = BlasUtils<Device::Type::kKunlun>::GetDataType(b_type_);
    auto dt_b = BlasUtils<Device::Type::kKunlun>::GetDataType(a_type_);
    auto dt_c = c_is_f32
                    ? BL::R_32F
                    : BlasUtils<Device::Type::kKunlun>::GetDataType(y_type_);
    auto compute =
        BlasUtils<Device::Type::kKunlun>::GetComputeType(DataType::kFloat32);

    const float alpha = alpha_;
    const float beta = 0.0f;         // bias up-cast handled by the caller.
    const int mm = (int)y_size_m();  // y.size(-2)
    const int nn = (int)y_size_n();  // y.size(-1)
    const int kk = (int)k_;
    const int lda = lda_b;
    const int ldb = (int)a.stride(-2);
    const int ldc = (int)ldy_;
    const long long sa = batch_stride_b_;
    const long long sb = batch_stride_a_;
    const long long sc = batch_stride_y_;

    static const bool kDump = [] {
      const char* e = getenv("INFINI_KL_DUMP");
      return e && e[0] == '1';
    }();
    if (kDump) {
      static int seq = 0;
      int id = seq++;
      if (id < 3 || n_ == 152064) {
        std::string pa = "/tmp/kl_dump_" + std::to_string(id) + "_a.bin";
        std::string pb = "/tmp/kl_dump_" + std::to_string(id) + "_b.bin";
        std::string pm = "/tmp/kl_dump_" + std::to_string(id) + "_meta.txt";
        FILE* f = fopen(pm.c_str(), "w");
        fprintf(f,
                "m=%d n=%d k=%d lda=%d ldb=%d ldc=%d sa=%lld sb=%lld sc=%lld "
                "dt=%d batch=%d opa=%d opb=%d\n",
                nn, mm, kk, lda, ldb, ldc, (long long)sa, (long long)sb,
                (long long)sc, (int)y_type_, (int)batch_count_, (int)op_a,
                (int)op_b);
        fclose(f);
        {
          FILE* fa = fopen(pa.c_str(), "wb");
          std::vector<char> buf(((size_t)lda * (kk - 1) + nn) * 2);
          FILE* fb = fopen(pb.c_str(), "wb");
          std::vector<char> buf2(((size_t)ldb * (kk - 1) + mm) * 2);
          xpu_wait(static_cast<XPUStream>(stream_));
          xpu_memcpy_async(buf.data(), b.data(), buf.size(), XPU_DEVICE_TO_HOST,
                           static_cast<XPUStream>(stream_));
          xpu_memcpy_async(buf2.data(), a.data(), buf2.size(),
                           XPU_DEVICE_TO_HOST, static_cast<XPUStream>(stream_));
          xpu_wait(static_cast<XPUStream>(stream_));
          fwrite(buf.data(), 1, buf.size(), fa);
          fclose(fa);
          fwrite(buf2.data(), 1, buf2.size(), fb);
          fclose(fb);
        }
      }
    }

    if (y_type_ == DataType::kBFloat16) {
      // Strided-batched BLAS rejects bf16 entirely; loop GemmEx.
      const auto* b_base = static_cast<const char*>(b.data());
      const auto* a_base = static_cast<const char*>(a.data());
      if (getenv("INFINI_KL_PTRDUMP") != nullptr) {
        fprintf(stderr,
                "[klptr] batch=%d sa=%lld sb=%lld sc=%lld nn=%d mm=%d kk=%d "
                "lda=%d ldb=%d ldc=%d\n",
                (int)batch_count_, sa, sb, sc, nn, mm, kk, lda, ldb, ldc);
        for (int i = 0; i < (int)batch_count_; ++i) {
          fprintf(stderr, "[klptr] batch %d: A=%p B=%p C=%p\n", i,
                  (const void*)(b_base + (size_t)sa * 2 * (size_t)i),
                  (const void*)(a_base + (size_t)sb * 2 * (size_t)i),
                  (const void*)((char*)y_out + (size_t)sc * 4 * (size_t)i));
        }
      }
      for (int i = 0; i < (int)batch_count_; ++i) {
        auto status = BL::BlasGemmEx(
            handle, op_a, op_b, nn, mm, kk, &alpha,
            const_cast<char*>(const_cast<const char*>(b_base) +
                              (size_t)sa * 2 * (size_t)i),
            dt_a, lda,
            const_cast<char*>(const_cast<const char*>(a_base) +
                              (size_t)sb * 2 * (size_t)i),
            dt_b, ldb, &beta, (char*)y_out + (size_t)sc * 4 * (size_t)i, dt_c,
            ldc, compute, BL::BLAS_GEMM_DEFAULT);
        if (status != 0) {
          throw std::runtime_error("kunlun bf16 GEMM (GemmEx, batch " +
                                   std::to_string(i) +
                                   ") failed: " + std::to_string((int)status));
        }
      }
      return;
    }
    if (getenv("INFINI_KL_PTRDUMP") != nullptr) {
      fprintf(stderr,
              "[klptr-sb] batch=%d sa=%lld sb=%lld sc=%lld nn=%d mm=%d kk=%d "
              "lda=%d ldb=%d ldc=%d opa=%d opb=%d\n",
              (int)batch_count_, sa, sb, sc, nn, mm, kk, lda, ldb, ldc,
              (int)op_a, (int)op_b);
    }
    {
      static const bool kDumpSb = getenv("INFINI_KL_DUMP") != nullptr;
      if (kDumpSb && n_ == 152064) {
        static int lmid = 0;
        int id = 3 + (lmid++);
        const size_t esz = y_type_ == DataType::kFloat32
                               ? 4
                               : (y_type_ == DataType::kFloat64 ? 8 : 2);
        std::vector<char> buf(((size_t)lda * (kk - 1) + nn) * esz);
        std::vector<char> buf2(((size_t)ldb * (kk - 1) + mm) * esz);
        xpu_wait(static_cast<XPUStream>(stream_));
        xpu_memcpy_async(buf.data(), b.data(), buf.size(), XPU_DEVICE_TO_HOST,
                         static_cast<XPUStream>(stream_));
        xpu_memcpy_async(buf2.data(), a.data(), buf2.size(), XPU_DEVICE_TO_HOST,
                         static_cast<XPUStream>(stream_));
        xpu_wait(static_cast<XPUStream>(stream_));
        FILE* fa = fopen(
            ("/tmp/kl_dump_" + std::to_string(id) + "_a.bin").c_str(), "wb");
        fwrite(buf.data(), 1, buf.size(), fa);
        fclose(fa);
        FILE* fb = fopen(
            ("/tmp/kl_dump_" + std::to_string(id) + "_b.bin").c_str(), "wb");
        fwrite(buf2.data(), 1, buf2.size(), fb);
        fclose(fb);
        FILE* fm = fopen(
            ("/tmp/kl_dump_" + std::to_string(id) + "_meta.txt").c_str(), "w");
        fprintf(fm,
                "m=%d n=%d k=%d lda=%d ldb=%d ldc=%d dt=%d batch=%d opa=%d "
                "opb=%d\n",
                nn, mm, kk, lda, ldb, ldc, (int)y_type_, (int)batch_count_,
                (int)op_a, (int)op_b);
        fclose(fm);
      }
    }
    auto status = BL::BlasGemmStridedBatchedEx(
        handle, op_a, op_b, nn, mm, kk, &alpha, b.data(), dt_a, lda, sa,
        a.data(), dt_b, ldb, sb, &beta, y_out, dt_c, ldc, sc, (int)batch_count_,
        compute, BL::BLAS_GEMM_DEFAULT);
    if (status != 0) {
      throw std::runtime_error("kunlun GEMM failed: " +
                               std::to_string((int)status));
    }
  }

  std::size_t y_size_m() const { return m_; }
  std::size_t y_size_n() const { return n_; }
};

}  // namespace infini::ops

#endif
