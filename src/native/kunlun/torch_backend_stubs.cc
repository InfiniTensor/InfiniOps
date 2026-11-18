// Kunlun link-level stubs for the ATen (torch) backend (implementation
// slot 8), both generated (generated/torch/...) and hand-written
// (src/torch/ops/...) providers.
//
// The device-generic template `Operator<Op, kDev, 8> : public Op` makes
// slot 8 an active implementation for every device; only the devices
// listed in torch/instantiate.h get explicit instantiation definitions.
// This file provides throwing member-specialization definitions for
// kKunlun so libinfiniops links; calling one throws.
//
// All specializations are declared at namespace infini::ops scope using
// the class's qualified name - matching where the class template lives.

#include <stdexcept>

#include "operator.h"
#include "torch/abs/abs.h"
#include "torch/absolute/absolute.h"
#include "torch/acos/acos.h"
#include "torch/acosh/acosh.h"
#include "torch/adaptive_avg_pool2d/adaptive_avg_pool2d.h"
#include "torch/adaptive_avg_pool3d/adaptive_avg_pool3d.h"
#include "torch/adaptive_avg_pool3d_backward/adaptive_avg_pool3d_backward.h"
#include "torch/adaptive_max_pool2d/adaptive_max_pool2d.h"
#include "torch/adaptive_max_pool2d_backward/adaptive_max_pool2d_backward.h"
#include "torch/adaptive_max_pool3d/adaptive_max_pool3d.h"
#include "torch/adaptive_max_pool3d_backward/adaptive_max_pool3d_backward.h"
#include "torch/add/add.h"
#include "torch/addbmm/addbmm.h"
#include "torch/addcdiv/addcdiv.h"
#include "torch/addcmul/addcmul.h"
#include "torch/addmm/addmm.h"
#include "torch/addmv/addmv.h"
#include "torch/addr/addr.h"
#include "torch/all/all.h"
#include "torch/amax/amax.h"
#include "torch/amin/amin.h"
#include "torch/aminmax/aminmax.h"
#include "torch/angle/angle.h"
#include "torch/any/any.h"
#include "torch/arccos/arccos.h"
#include "torch/arccosh/arccosh.h"
#include "torch/arcsin/arcsin.h"
#include "torch/arcsinh/arcsinh.h"
#include "torch/arctan/arctan.h"
#include "torch/arctan2/arctan2.h"
#include "torch/arctanh/arctanh.h"
#include "torch/argmax/argmax.h"
#include "torch/argmin/argmin.h"
#include "torch/argsort/argsort.h"
#include "torch/asin/asin.h"
#include "torch/asinh/asinh.h"
#include "torch/atan/atan.h"
#include "torch/atan2/atan2.h"
#include "torch/atanh/atanh.h"
#include "torch/avg_pool2d/avg_pool2d.h"
#include "torch/avg_pool2d_backward/avg_pool2d_backward.h"
#include "torch/avg_pool3d/avg_pool3d.h"
#include "torch/avg_pool3d_backward/avg_pool3d_backward.h"
#include "torch/baddbmm/baddbmm.h"
#include "torch/batch_norm_elemt/batch_norm_elemt.h"
#include "torch/bernoulli/bernoulli.h"
#include "torch/binary_cross_entropy/binary_cross_entropy.h"
#include "torch/binary_cross_entropy_backward/binary_cross_entropy_backward.h"
#include "torch/bitwise_and/bitwise_and.h"
#include "torch/bitwise_left_shift/bitwise_left_shift.h"
#include "torch/bitwise_not/bitwise_not.h"
#include "torch/bitwise_or/bitwise_or.h"
#include "torch/bitwise_right_shift/bitwise_right_shift.h"
#include "torch/bitwise_xor/bitwise_xor.h"
#include "torch/bmm/bmm.h"
#include "torch/bucketize/bucketize.h"
#include "torch/cat/cat.h"
#include "torch/ceil/ceil.h"
#include "torch/chain_matmul/chain_matmul.h"
#include "torch/cholesky/cholesky.h"
#include "torch/cholesky_inverse/cholesky_inverse.h"
#include "torch/cholesky_solve/cholesky_solve.h"
#include "torch/clamp/clamp.h"
#include "torch/clamp_max/clamp_max.h"
#include "torch/clamp_min/clamp_min.h"
#include "torch/clip/clip.h"
#include "torch/col2im/col2im.h"
#include "torch/column_stack/column_stack.h"
#include "torch/complex/complex.h"
#include "torch/concatenate/concatenate.h"
#include "torch/conj_physical/conj_physical.h"
#include "torch/copysign/copysign.h"
#include "torch/cos/cos.h"
#include "torch/cosh/cosh.h"
#include "torch/cross/cross.h"
#include "torch/cudnn_convolution/cudnn_convolution.h"
#include "torch/cummax/cummax.h"
#include "torch/cummin/cummin.h"
#include "torch/cumprod/cumprod.h"
#include "torch/cumsum/cumsum.h"
#include "torch/deg2rad/deg2rad.h"
#include "torch/diag/diag.h"
#include "torch/diff/diff.h"
#include "torch/digamma/digamma.h"
#include "torch/div/div.h"
#include "torch/divide/divide.h"
#include "torch/dot/dot.h"
#include "torch/dstack/dstack.h"
#include "torch/elu/elu.h"
#include "torch/elu_backward/elu_backward.h"
#include "torch/eq/eq.h"
#include "torch/erf/erf.h"
#include "torch/erfc/erfc.h"
#include "torch/erfinv/erfinv.h"
#include "torch/exp/exp.h"
#include "torch/exp2/exp2.h"
#include "torch/expm1/expm1.h"
#include "torch/fft_fft/fft_fft.h"
#include "torch/fft_fft2/fft_fft2.h"
#include "torch/fft_fftn/fft_fftn.h"
#include "torch/fft_hfft/fft_hfft.h"
#include "torch/fft_hfft2/fft_hfft2.h"
#include "torch/fft_hfftn/fft_hfftn.h"
#include "torch/fft_ifft/fft_ifft.h"
#include "torch/fft_ifft2/fft_ifft2.h"
#include "torch/fft_ifftn/fft_ifftn.h"
#include "torch/fft_ihfft/fft_ihfft.h"
#include "torch/fft_ihfft2/fft_ihfft2.h"
#include "torch/fft_ihfftn/fft_ihfftn.h"
#include "torch/fft_irfft/fft_irfft.h"
#include "torch/fft_irfft2/fft_irfft2.h"
#include "torch/fft_irfftn/fft_irfftn.h"
#include "torch/fft_rfft/fft_rfft.h"
#include "torch/fft_rfft2/fft_rfft2.h"
#include "torch/fft_rfftn/fft_rfftn.h"
#include "torch/fill/fill.h"
#include "torch/fix/fix.h"
#include "torch/float_power/float_power.h"
#include "torch/floor/floor.h"
#include "torch/floor_divide/floor_divide.h"
#include "torch/fmax/fmax.h"
#include "torch/fmin/fmin.h"
#include "torch/fmod/fmod.h"
#include "torch/frac/frac.h"
#include "torch/fractional_max_pool2d/fractional_max_pool2d.h"
#include "torch/fractional_max_pool2d_backward/fractional_max_pool2d_backward.h"
#include "torch/fractional_max_pool3d/fractional_max_pool3d.h"
#include "torch/fractional_max_pool3d_backward/fractional_max_pool3d_backward.h"
#include "torch/frexp/frexp.h"
#include "torch/frobenius_norm/frobenius_norm.h"
#include "torch/gather/gather.h"
#include "torch/gcd/gcd.h"
#include "torch/ge/ge.h"
#include "torch/gelu/gelu.h"
#include "torch/gelu_backward/gelu_backward.h"
#include "torch/geqrf/geqrf.h"
#include "torch/ger/ger.h"
#include "torch/glu/glu.h"
#include "torch/glu_backward/glu_backward.h"
#include "torch/greater/greater.h"
#include "torch/greater_equal/greater_equal.h"
#include "torch/gt/gt.h"
#include "torch/hardshrink/hardshrink.h"
#include "torch/hardshrink_backward/hardshrink_backward.h"
#include "torch/hardsigmoid/hardsigmoid.h"
#include "torch/hardsigmoid_backward/hardsigmoid_backward.h"
#include "torch/hardswish/hardswish.h"
#include "torch/hardtanh/hardtanh.h"
#include "torch/hardtanh_backward/hardtanh_backward.h"
#include "torch/heaviside/heaviside.h"
#include "torch/histc/histc.h"
#include "torch/histogram/histogram.h"
#include "torch/hspmm/hspmm.h"
#include "torch/hstack/hstack.h"
#include "torch/huber_loss/huber_loss.h"
#include "torch/huber_loss_backward/huber_loss_backward.h"
#include "torch/hypot/hypot.h"
#include "torch/i0/i0.h"
#include "torch/igamma/igamma.h"
#include "torch/igammac/igammac.h"
#include "torch/im2col/im2col.h"
#include "torch/index/index.h"
#include "torch/index_add/index_add.h"
#include "torch/index_copy/index_copy.h"
#include "torch/index_fill/index_fill.h"
#include "torch/index_reduce/index_reduce.h"
#include "torch/index_select/index_select.h"
#include "torch/inner/inner.h"
#include "torch/internal_add_relu/internal_add_relu.h"
#include "torch/internal_addmm_activation/internal_addmm_activation.h"
#include "torch/internal_batch_norm_with_update/internal_batch_norm_with_update.h"
#include "torch/internal_chunk_cat/internal_chunk_cat.h"
#include "torch/internal_compute_linear_combination/internal_compute_linear_combination.h"
#include "torch/internal_conv_depthwise2d/internal_conv_depthwise2d.h"
#include "torch/internal_convert_indices_from_coo_to_csr/internal_convert_indices_from_coo_to_csr.h"
#include "torch/internal_convert_indices_from_csr_to_coo/internal_convert_indices_from_csr_to_coo.h"
#include "torch/internal_fft_c2c/internal_fft_c2c.h"
#include "torch/internal_fft_c2r/internal_fft_c2r.h"
#include "torch/internal_fft_r2c/internal_fft_r2c.h"
#include "torch/internal_int_mm/internal_int_mm.h"
#include "torch/internal_linalg_det/internal_linalg_det.h"
#include "torch/internal_linalg_eigh/internal_linalg_eigh.h"
#include "torch/internal_linalg_slogdet/internal_linalg_slogdet.h"
#include "torch/internal_linalg_solve_ex/internal_linalg_solve_ex.h"
#include "torch/internal_linalg_svd/internal_linalg_svd.h"
#include "torch/internal_log_softmax/internal_log_softmax.h"
#include "torch/internal_log_softmax_backward_data/internal_log_softmax_backward_data.h"
#include "torch/internal_logcumsumexp/internal_logcumsumexp.h"
#include "torch/internal_native_batch_norm_legit/internal_native_batch_norm_legit.h"
#include "torch/internal_scaled_mm/internal_scaled_mm.h"
#include "torch/internal_slow_conv2d_backward/internal_slow_conv2d_backward.h"
#include "torch/internal_slow_conv2d_forward/internal_slow_conv2d_forward.h"
#include "torch/internal_softmax/internal_softmax.h"
#include "torch/internal_softmax_backward_data/internal_softmax_backward_data.h"
#include "torch/internal_stack/internal_stack.h"
#include "torch/internal_upsample_bicubic2d_aa/internal_upsample_bicubic2d_aa.h"
#include "torch/internal_upsample_bicubic2d_aa_backward/internal_upsample_bicubic2d_aa_backward.h"
#include "torch/internal_upsample_bilinear2d_aa/internal_upsample_bilinear2d_aa.h"
#include "torch/internal_upsample_bilinear2d_aa_backward/internal_upsample_bilinear2d_aa_backward.h"
#include "torch/internal_upsample_nearest_exact1d/internal_upsample_nearest_exact1d.h"
#include "torch/internal_upsample_nearest_exact1d_backward/internal_upsample_nearest_exact1d_backward.h"
#include "torch/internal_upsample_nearest_exact2d/internal_upsample_nearest_exact2d.h"
#include "torch/internal_upsample_nearest_exact2d_backward/internal_upsample_nearest_exact2d_backward.h"
#include "torch/internal_upsample_nearest_exact3d/internal_upsample_nearest_exact3d.h"
#include "torch/internal_upsample_nearest_exact3d_backward/internal_upsample_nearest_exact3d_backward.h"
#include "torch/inverse/inverse.h"
#include "torch/isin/isin.h"
#include "torch/isneginf/isneginf.h"
#include "torch/isposinf/isposinf.h"
#include "torch/kron/kron.h"
#include "torch/kthvalue/kthvalue.h"
#include "torch/lcm/lcm.h"
#include "torch/ldexp/ldexp.h"
#include "torch/le/le.h"
#include "torch/leaky_relu/leaky_relu.h"
#include "torch/leaky_relu_backward/leaky_relu_backward.h"
#include "torch/lerp/lerp.h"
#include "torch/less/less.h"
#include "torch/less_equal/less_equal.h"
#include "torch/lgamma/lgamma.h"
#include "torch/linalg_cholesky/linalg_cholesky.h"
#include "torch/linalg_cholesky_ex/linalg_cholesky_ex.h"
#include "torch/linalg_cond/linalg_cond.h"
#include "torch/linalg_cross/linalg_cross.h"
#include "torch/linalg_det/linalg_det.h"
#include "torch/linalg_eig/linalg_eig.h"
#include "torch/linalg_eigh/linalg_eigh.h"
#include "torch/linalg_eigvals/linalg_eigvals.h"
#include "torch/linalg_eigvalsh/linalg_eigvalsh.h"
#include "torch/linalg_householder_product/linalg_householder_product.h"
#include "torch/linalg_inv/linalg_inv.h"
#include "torch/linalg_inv_ex/linalg_inv_ex.h"
#include "torch/linalg_ldl_factor/linalg_ldl_factor.h"
#include "torch/linalg_ldl_factor_ex/linalg_ldl_factor_ex.h"
#include "torch/linalg_ldl_solve/linalg_ldl_solve.h"
#include "torch/linalg_lstsq/linalg_lstsq.h"
#include "torch/linalg_lu/linalg_lu.h"
#include "torch/linalg_lu_factor/linalg_lu_factor.h"
#include "torch/linalg_lu_factor_ex/linalg_lu_factor_ex.h"
#include "torch/linalg_lu_solve/linalg_lu_solve.h"
#include "torch/linalg_matmul/linalg_matmul.h"
#include "torch/linalg_matrix_norm/linalg_matrix_norm.h"
#include "torch/linalg_matrix_power/linalg_matrix_power.h"
#include "torch/linalg_matrix_rank/linalg_matrix_rank.h"
#include "torch/linalg_multi_dot/linalg_multi_dot.h"
#include "torch/linalg_norm/linalg_norm.h"
#include "torch/linalg_pinv/linalg_pinv.h"
#include "torch/linalg_qr/linalg_qr.h"
#include "torch/linalg_slogdet/linalg_slogdet.h"
#include "torch/linalg_solve/linalg_solve.h"
#include "torch/linalg_solve_ex/linalg_solve_ex.h"
#include "torch/linalg_solve_triangular/linalg_solve_triangular.h"
#include "torch/linalg_svd/linalg_svd.h"
#include "torch/linalg_svdvals/linalg_svdvals.h"
#include "torch/linalg_tensorinv/linalg_tensorinv.h"
#include "torch/linalg_tensorsolve/linalg_tensorsolve.h"
#include "torch/linalg_vecdot/linalg_vecdot.h"
#include "torch/linalg_vector_norm/linalg_vector_norm.h"
#include "torch/linear/linear.h"
#include "torch/linspace/linspace.h"
#include "torch/log/log.h"
#include "torch/log10/log10.h"
#include "torch/log1p/log1p.h"
#include "torch/log2/log2.h"
#include "torch/log_sigmoid/log_sigmoid.h"
#include "torch/log_sigmoid_backward/log_sigmoid_backward.h"
#include "torch/log_sigmoid_forward/log_sigmoid_forward.h"
#include "torch/log_softmax/log_softmax.h"
#include "torch/logaddexp/logaddexp.h"
#include "torch/logaddexp2/logaddexp2.h"
#include "torch/logcumsumexp/logcumsumexp.h"
#include "torch/logical_and/logical_and.h"
#include "torch/logical_not/logical_not.h"
#include "torch/logical_or/logical_or.h"
#include "torch/logical_xor/logical_xor.h"
#include "torch/logit/logit.h"
#include "torch/logit_backward/logit_backward.h"
#include "torch/logspace/logspace.h"
#include "torch/logsumexp/logsumexp.h"
#include "torch/lt/lt.h"
#include "torch/lu_solve/lu_solve.h"
#include "torch/lu_unpack/lu_unpack.h"
#include "torch/masked_fill/masked_fill.h"
#include "torch/masked_select/masked_select.h"
#include "torch/matmul/matmul.h"
#include "torch/matrix_power/matrix_power.h"
#include "torch/max/max.h"
#include "torch/max_pool2d_with_indices/max_pool2d_with_indices.h"
#include "torch/max_pool2d_with_indices_backward/max_pool2d_with_indices_backward.h"
#include "torch/max_pool3d_with_indices/max_pool3d_with_indices.h"
#include "torch/max_pool3d_with_indices_backward/max_pool3d_with_indices_backward.h"
#include "torch/max_unpool2d/max_unpool2d.h"
#include "torch/max_unpool3d/max_unpool3d.h"
#include "torch/maximum/maximum.h"
#include "torch/mean/mean.h"
#include "torch/median/median.h"
#include "torch/min/min.h"
#include "torch/minimum/minimum.h"
#include "torch/mish/mish.h"
#include "torch/mkldnn_adaptive_avg_pool2d/mkldnn_adaptive_avg_pool2d.h"
#include "torch/mm/mm.h"
#include "torch/mode/mode.h"
#include "torch/mse_loss/mse_loss.h"
#include "torch/mse_loss_backward/mse_loss_backward.h"
#include "torch/msort/msort.h"
#include "torch/mul/mul.h"
#include "torch/multi_margin_loss/multi_margin_loss.h"
#include "torch/multi_margin_loss_backward/multi_margin_loss_backward.h"
#include "torch/multilabel_margin_loss/multilabel_margin_loss.h"
#include "torch/multilabel_margin_loss_backward/multilabel_margin_loss_backward.h"
#include "torch/multilabel_margin_loss_forward/multilabel_margin_loss_forward.h"
#include "torch/multinomial/multinomial.h"
#include "torch/multiply/multiply.h"
#include "torch/mv/mv.h"
#include "torch/mvlgamma/mvlgamma.h"
#include "torch/nan_to_num/nan_to_num.h"
#include "torch/nanmean/nanmean.h"
#include "torch/nanmedian/nanmedian.h"
#include "torch/nanquantile/nanquantile.h"
#include "torch/nansum/nansum.h"
#include "torch/narrow_copy/narrow_copy.h"
#include "torch/native_batch_norm/native_batch_norm.h"
#include "torch/native_layer_norm/native_layer_norm.h"
#include "torch/ne/ne.h"
#include "torch/neg/neg.h"
#include "torch/negative/negative.h"
#include "torch/nextafter/nextafter.h"
#include "torch/nll_loss/nll_loss.h"
#include "torch/nll_loss2d/nll_loss2d.h"
#include "torch/nll_loss2d_backward/nll_loss2d_backward.h"
#include "torch/nll_loss2d_forward/nll_loss2d_forward.h"
#include "torch/nll_loss_backward/nll_loss_backward.h"
#include "torch/nll_loss_forward/nll_loss_forward.h"
#include "torch/nonzero/nonzero.h"
#include "torch/nonzero_static/nonzero_static.h"
#include "torch/norm/norm.h"
#include "torch/normal/normal.h"
#include "torch/not_equal/not_equal.h"
#include "torch/nuclear_norm/nuclear_norm.h"
#include "torch/ops/max_pool2d/max_pool2d.h"
#include "torch/orgqr/orgqr.h"
#include "torch/ormqr/ormqr.h"
#include "torch/outer/outer.h"
#include "torch/polar/polar.h"
#include "torch/pow/pow.h"
#include "torch/prod/prod.h"
#include "torch/qr/qr.h"
#include "torch/quantile/quantile.h"
#include "torch/rad2deg/rad2deg.h"
#include "torch/random/random.h"
#include "torch/reciprocal/reciprocal.h"
#include "torch/reflection_pad1d/reflection_pad1d.h"
#include "torch/reflection_pad1d_backward/reflection_pad1d_backward.h"
#include "torch/reflection_pad2d/reflection_pad2d.h"
#include "torch/reflection_pad2d_backward/reflection_pad2d_backward.h"
#include "torch/reflection_pad3d/reflection_pad3d.h"
#include "torch/reflection_pad3d_backward/reflection_pad3d_backward.h"
#include "torch/relu/relu.h"
#include "torch/remainder/remainder.h"
#include "torch/renorm/renorm.h"
#include "torch/replication_pad1d/replication_pad1d.h"
#include "torch/replication_pad1d_backward/replication_pad1d_backward.h"
#include "torch/replication_pad2d/replication_pad2d.h"
#include "torch/replication_pad2d_backward/replication_pad2d_backward.h"
#include "torch/replication_pad3d/replication_pad3d.h"
#include "torch/replication_pad3d_backward/replication_pad3d_backward.h"
#include "torch/round/round.h"
#include "torch/row_stack/row_stack.h"
#include "torch/rrelu_with_noise/rrelu_with_noise.h"
#include "torch/rsqrt/rsqrt.h"
#include "torch/scatter/scatter.h"
#include "torch/scatter_add/scatter_add.h"
#include "torch/scatter_reduce/scatter_reduce.h"
#include "torch/searchsorted/searchsorted.h"
#include "torch/set/set.h"
#include "torch/sgn/sgn.h"
#include "torch/sigmoid/sigmoid.h"
#include "torch/sigmoid_backward/sigmoid_backward.h"
#include "torch/sign/sign.h"
#include "torch/signbit/signbit.h"
#include "torch/silu/silu.h"
#include "torch/silu_backward/silu_backward.h"
#include "torch/sin/sin.h"
#include "torch/sinc/sinc.h"
#include "torch/sinh/sinh.h"
#include "torch/slogdet/slogdet.h"
#include "torch/slow_conv3d/slow_conv3d.h"
#include "torch/slow_conv3d_forward/slow_conv3d_forward.h"
#include "torch/slow_conv_transpose2d/slow_conv_transpose2d.h"
#include "torch/slow_conv_transpose3d/slow_conv_transpose3d.h"
#include "torch/smooth_l1_loss/smooth_l1_loss.h"
#include "torch/smooth_l1_loss_backward/smooth_l1_loss_backward.h"
#include "torch/soft_margin_loss/soft_margin_loss.h"
#include "torch/soft_margin_loss_backward/soft_margin_loss_backward.h"
#include "torch/softmax/softmax.h"
#include "torch/softplus/softplus.h"
#include "torch/softplus_backward/softplus_backward.h"
#include "torch/softshrink/softshrink.h"
#include "torch/softshrink_backward/softshrink_backward.h"
#include "torch/sort/sort.h"
#include "torch/sparse_sampled_addmm/sparse_sampled_addmm.h"
#include "torch/special_airy_ai/special_airy_ai.h"
#include "torch/special_bessel_j0/special_bessel_j0.h"
#include "torch/special_bessel_j1/special_bessel_j1.h"
#include "torch/special_bessel_y0/special_bessel_y0.h"
#include "torch/special_bessel_y1/special_bessel_y1.h"
#include "torch/special_chebyshev_polynomial_t/special_chebyshev_polynomial_t.h"
#include "torch/special_chebyshev_polynomial_u/special_chebyshev_polynomial_u.h"
#include "torch/special_chebyshev_polynomial_v/special_chebyshev_polynomial_v.h"
#include "torch/special_chebyshev_polynomial_w/special_chebyshev_polynomial_w.h"
#include "torch/special_digamma/special_digamma.h"
#include "torch/special_entr/special_entr.h"
#include "torch/special_erf/special_erf.h"
#include "torch/special_erfc/special_erfc.h"
#include "torch/special_erfcx/special_erfcx.h"
#include "torch/special_erfinv/special_erfinv.h"
#include "torch/special_exp2/special_exp2.h"
#include "torch/special_expit/special_expit.h"
#include "torch/special_expm1/special_expm1.h"
#include "torch/special_gammainc/special_gammainc.h"
#include "torch/special_gammaincc/special_gammaincc.h"
#include "torch/special_gammaln/special_gammaln.h"
#include "torch/special_hermite_polynomial_h/special_hermite_polynomial_h.h"
#include "torch/special_hermite_polynomial_he/special_hermite_polynomial_he.h"
#include "torch/special_i0/special_i0.h"
#include "torch/special_i0e/special_i0e.h"
#include "torch/special_i1/special_i1.h"
#include "torch/special_i1e/special_i1e.h"
#include "torch/special_laguerre_polynomial_l/special_laguerre_polynomial_l.h"
#include "torch/special_legendre_polynomial_p/special_legendre_polynomial_p.h"
#include "torch/special_log1p/special_log1p.h"
#include "torch/special_log_ndtr/special_log_ndtr.h"
#include "torch/special_logit/special_logit.h"
#include "torch/special_logsumexp/special_logsumexp.h"
#include "torch/special_modified_bessel_i0/special_modified_bessel_i0.h"
#include "torch/special_modified_bessel_i1/special_modified_bessel_i1.h"
#include "torch/special_modified_bessel_k0/special_modified_bessel_k0.h"
#include "torch/special_modified_bessel_k1/special_modified_bessel_k1.h"
#include "torch/special_multigammaln/special_multigammaln.h"
#include "torch/special_ndtr/special_ndtr.h"
#include "torch/special_ndtri/special_ndtri.h"
#include "torch/special_psi/special_psi.h"
#include "torch/special_round/special_round.h"
#include "torch/special_scaled_modified_bessel_k0/special_scaled_modified_bessel_k0.h"
#include "torch/special_scaled_modified_bessel_k1/special_scaled_modified_bessel_k1.h"
#include "torch/special_shifted_chebyshev_polynomial_t/special_shifted_chebyshev_polynomial_t.h"
#include "torch/special_shifted_chebyshev_polynomial_u/special_shifted_chebyshev_polynomial_u.h"
#include "torch/special_shifted_chebyshev_polynomial_v/special_shifted_chebyshev_polynomial_v.h"
#include "torch/special_shifted_chebyshev_polynomial_w/special_shifted_chebyshev_polynomial_w.h"
#include "torch/special_sinc/special_sinc.h"
#include "torch/special_spherical_bessel_j0/special_spherical_bessel_j0.h"
#include "torch/special_xlog1py/special_xlog1py.h"
#include "torch/special_xlogy/special_xlogy.h"
#include "torch/special_zeta/special_zeta.h"
#include "torch/sqrt/sqrt.h"
#include "torch/square/square.h"
#include "torch/sspaddmm/sspaddmm.h"
#include "torch/stack/stack.h"
#include "torch/std/std.h"
#include "torch/sub/sub.h"
#include "torch/subtract/subtract.h"
#include "torch/sum/sum.h"
#include "torch/svd/svd.h"
#include "torch/take/take.h"
#include "torch/take_along_dim/take_along_dim.h"
#include "torch/tan/tan.h"
#include "torch/tanh/tanh.h"
#include "torch/tanh_backward/tanh_backward.h"
#include "torch/tensordot/tensordot.h"
#include "torch/thnn_conv2d/thnn_conv2d.h"
#include "torch/threshold/threshold.h"
#include "torch/threshold_backward/threshold_backward.h"
#include "torch/topk/topk.h"
#include "torch/triangular_solve/triangular_solve.h"
#include "torch/tril/tril.h"
#include "torch/triu/triu.h"
#include "torch/true_divide/true_divide.h"
#include "torch/trunc/trunc.h"
#include "torch/upsample_bicubic2d/upsample_bicubic2d.h"
#include "torch/upsample_bicubic2d_backward/upsample_bicubic2d_backward.h"
#include "torch/upsample_bilinear2d/upsample_bilinear2d.h"
#include "torch/upsample_bilinear2d_backward/upsample_bilinear2d_backward.h"
#include "torch/upsample_linear1d/upsample_linear1d.h"
#include "torch/upsample_linear1d_backward/upsample_linear1d_backward.h"
#include "torch/upsample_nearest1d/upsample_nearest1d.h"
#include "torch/upsample_nearest1d_backward/upsample_nearest1d_backward.h"
#include "torch/upsample_nearest2d/upsample_nearest2d.h"
#include "torch/upsample_nearest2d_backward/upsample_nearest2d_backward.h"
#include "torch/upsample_nearest3d/upsample_nearest3d.h"
#include "torch/upsample_nearest3d_backward/upsample_nearest3d_backward.h"
#include "torch/upsample_trilinear3d/upsample_trilinear3d.h"
#include "torch/upsample_trilinear3d_backward/upsample_trilinear3d_backward.h"
#include "torch/var/var.h"
#include "torch/vdot/vdot.h"
#include "torch/vstack/vstack.h"
#include "torch/where/where.h"
#include "torch/xlogy/xlogy.h"

namespace infini::ops {

template <>
void Operator<Abs, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Abs (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Absolute, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Absolute (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Acos, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Acos (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Acosh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Acosh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveAvgPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    Tensor out) const {
  throw std::runtime_error(
      "AdaptiveAvgPool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveAvgPool3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    Tensor out) const {
  throw std::runtime_error(
      "AdaptiveAvgPool3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveAvgPool3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, Tensor grad_input) const {
  throw std::runtime_error(
      "AdaptiveAvgPool3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveMaxPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size, Tensor out,
    Tensor indices) const {
  throw std::runtime_error(
      "AdaptiveMaxPool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveMaxPool2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor indices,
    Tensor grad_input) const {
  throw std::runtime_error(
      "AdaptiveMaxPool2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveMaxPool3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size, Tensor out,
    Tensor indices) const {
  throw std::runtime_error(
      "AdaptiveMaxPool3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AdaptiveMaxPool3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor indices,
    Tensor grad_input) const {
  throw std::runtime_error(
      "AdaptiveMaxPool3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Add, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                         const Tensor other,
                                                         const double alpha,
                                                         Tensor out) const {
  throw std::runtime_error("Add (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addbmm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor batch1, const Tensor batch2,
    const double beta, const double alpha, Tensor out) const {
  throw std::runtime_error("Addbmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addcdiv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor tensor1, const Tensor tensor2,
    const double value, Tensor out) const {
  throw std::runtime_error("Addcdiv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addcmul, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor tensor1, const Tensor tensor2,
    const double value, Tensor out) const {
  throw std::runtime_error("Addcmul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addmm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat1, const Tensor mat2, const double beta,
    const double alpha, Tensor out) const {
  throw std::runtime_error("Addmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addmv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat, const Tensor vec, const double beta,
    const double alpha, Tensor out) const {
  throw std::runtime_error("Addmv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Addr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor vec1, const Tensor vec2, const double beta,
    const double alpha, Tensor out) const {
  throw std::runtime_error("Addr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<All, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const int64_t dim,
                                                         const bool keepdim,
                                                         Tensor out) const {
  throw std::runtime_error("All (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<All, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, Tensor out) const {
  throw std::runtime_error("All (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<All, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("All (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Amax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Amax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Amin, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Amin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Aminmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> dim, const bool keepdim,
    Tensor min, Tensor max) const {
  throw std::runtime_error("Aminmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Angle, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Angle (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Any, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const int64_t dim,
                                                         const bool keepdim,
                                                         Tensor out) const {
  throw std::runtime_error("Any (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Any, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, Tensor out) const {
  throw std::runtime_error("Any (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Any, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Any (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arccos, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Arccos (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arccosh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Arccosh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arcsin, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Arcsin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arcsinh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Arcsinh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arctan, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Arctan (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arctan2, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor other,
                                                             Tensor out) const {
  throw std::runtime_error("Arctan2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Arctanh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Arctanh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Argmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Argmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Argmin, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Argmin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Argsort, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool stable, const int64_t dim,
    const bool descending, Tensor out) const {
  throw std::runtime_error("Argsort (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Asin, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Asin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Asinh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Asinh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Atan, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Atan (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Atan2, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor other,
                                                           Tensor out) const {
  throw std::runtime_error("Atan2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Atanh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Atanh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AvgPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const bool ceil_mode, const bool count_include_pad,
    const std::optional<int64_t> divisor_override, Tensor out) const {
  throw std::runtime_error(
      "AvgPool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AvgPool2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, const bool ceil_mode,
    const bool count_include_pad, const std::optional<int64_t> divisor_override,
    Tensor grad_input) const {
  throw std::runtime_error(
      "AvgPool2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AvgPool3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const bool ceil_mode, const bool count_include_pad,
    const std::optional<int64_t> divisor_override, Tensor out) const {
  throw std::runtime_error(
      "AvgPool3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<AvgPool3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, const bool ceil_mode,
    const bool count_include_pad, const std::optional<int64_t> divisor_override,
    Tensor grad_input) const {
  throw std::runtime_error(
      "AvgPool3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Baddbmm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor batch1, const Tensor batch2,
    const double beta, const double alpha, Tensor out) const {
  throw std::runtime_error("Baddbmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BatchNormElemt, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> weight,
    const std::optional<Tensor> bias, const Tensor mean, const Tensor invstd,
    const double eps, Tensor out) const {
  throw std::runtime_error(
      "BatchNormElemt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Bernoulli, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Bernoulli (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Bernoulli, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double p, Tensor out) const {
  throw std::runtime_error(
      "Bernoulli (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BinaryCrossEntropy, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const std::optional<Tensor> weight,
    const int64_t reduction, Tensor out) const {
  throw std::runtime_error(
      "BinaryCrossEntropy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BinaryCrossEntropyBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const std::optional<Tensor> weight, const int64_t reduction,
    Tensor grad_input) const {
  throw std::runtime_error(
      "BinaryCrossEntropyBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseAnd, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseAnd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseAnd, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseAnd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseAnd, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "BitwiseAnd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseAnd, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "BitwiseAnd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseLeftShift, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseLeftShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseLeftShift, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseLeftShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseLeftShift, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "BitwiseLeftShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseLeftShift, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "BitwiseLeftShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseNot, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "BitwiseNot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseOr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseOr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseOr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseOr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseOr, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "BitwiseOr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseOr, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "BitwiseOr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseRightShift, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseRightShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseRightShift, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseRightShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseRightShift, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "BitwiseRightShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseRightShift, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "BitwiseRightShift (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseXor, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseXor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseXor, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "BitwiseXor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseXor, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "BitwiseXor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<BitwiseXor, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "BitwiseXor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Bmm, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor mat2,
                                                         Tensor out) const {
  throw std::runtime_error("Bmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Bucketize, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor boundaries, const bool out_int32,
    const bool right, Tensor out) const {
  throw std::runtime_error(
      "Bucketize (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cat, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, const int64_t dim, Tensor out) const {
  throw std::runtime_error("Cat (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ceil, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Ceil (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ChainMatmul, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> matrices, Tensor out) const {
  throw std::runtime_error(
      "ChainMatmul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cholesky, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool upper, Tensor out) const {
  throw std::runtime_error(
      "Cholesky (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<CholeskyInverse, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool upper, Tensor out) const {
  throw std::runtime_error(
      "CholeskyInverse (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<CholeskySolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor input2, const bool upper,
    Tensor out) const {
  throw std::runtime_error(
      "CholeskySolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Clamp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> min,
    const std::optional<double> max, Tensor out) const {
  throw std::runtime_error("Clamp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Clamp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> min,
    const std::optional<Tensor> max, Tensor out) const {
  throw std::runtime_error("Clamp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double max, Tensor out) const {
  throw std::runtime_error(
      "ClampMax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor max, Tensor out) const {
  throw std::runtime_error(
      "ClampMax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMax, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor max) const {
  throw std::runtime_error(
      "ClampMax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMin, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double min, Tensor out) const {
  throw std::runtime_error(
      "ClampMin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMin, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor min, Tensor out) const {
  throw std::runtime_error(
      "ClampMin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ClampMin, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor min) const {
  throw std::runtime_error(
      "ClampMin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Clip, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> min,
    const std::optional<double> max, Tensor out) const {
  throw std::runtime_error("Clip (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Clip, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> min,
    const std::optional<Tensor> max, Tensor out) const {
  throw std::runtime_error("Clip (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Col2im, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> dilation,
    const std::vector<int64_t> padding, const std::vector<int64_t> stride,
    Tensor out) const {
  throw std::runtime_error("Col2im (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ColumnStack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error(
      "ColumnStack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Complex, Device::Type::kKunlun, 8>::operator()(const Tensor real,
                                                             const Tensor imag,
                                                             Tensor out) const {
  throw std::runtime_error("Complex (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Concatenate, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "Concatenate (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ConjPhysical, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "ConjPhysical (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Copysign, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Copysign (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Copysign, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "Copysign (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Copysign, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "Copysign (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Copysign, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "Copysign (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cos, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Cos (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cosh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Cosh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cross, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, const std::optional<int64_t> dim,
    Tensor out) const {
  throw std::runtime_error("Cross (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<CudnnConvolution, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::vector<int64_t> padding,
    const std::vector<int64_t> stride, const std::vector<int64_t> dilation,
    const int64_t groups, const bool benchmark, const bool deterministic,
    const bool allow_tf32, Tensor out) const {
  throw std::runtime_error(
      "CudnnConvolution (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cummax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, Tensor values,
    Tensor indices) const {
  throw std::runtime_error("Cummax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cummin, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, Tensor values,
    Tensor indices) const {
  throw std::runtime_error("Cummin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cumprod, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const std::optional<DataType> dtype,
    Tensor out) const {
  throw std::runtime_error("Cumprod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Cumsum, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const std::optional<DataType> dtype,
    Tensor out) const {
  throw std::runtime_error("Cumsum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Deg2rad, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Deg2rad (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Diag, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t diagonal, Tensor out) const {
  throw std::runtime_error("Diag (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Diff, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> prepend,
    const std::optional<Tensor> append, const int64_t n, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error("Diff (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Digamma, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Digamma (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other,
    const std::optional<std::string> rounding_mode, Tensor out) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other,
    const std::optional<std::string> rounding_mode) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const double other,
                                                         Tensor out) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Div, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other,
    const std::optional<std::string> rounding_mode, Tensor out) const {
  throw std::runtime_error("Div (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            const Tensor other,
                                                            Tensor out) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other,
    const std::optional<std::string> rounding_mode, Tensor out) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                            const double other,
                                                            Tensor out) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other,
    const std::optional<std::string> rounding_mode) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Divide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other,
    const std::optional<std::string> rounding_mode, Tensor out) const {
  throw std::runtime_error("Divide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Dot, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor tensor,
                                                         Tensor out) const {
  throw std::runtime_error("Dot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Dstack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error("Dstack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Elu, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double alpha, const double scale,
    const double input_scale, Tensor out) const {
  throw std::runtime_error("Elu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<EluBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor self_or_result, const double alpha,
    const double scale, const double input_scale, const bool is_result,
    Tensor grad_input) const {
  throw std::runtime_error(
      "EluBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Eq, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Eq (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Eq, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Eq (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Eq, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Eq (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Eq, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Eq (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Erf, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Erf (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Erfc, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Erfc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Erfinv, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Erfinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Exp, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Exp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Exp2, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Exp2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Expm1, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Expm1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Fft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Fft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Fft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Fft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Fftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Fftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Hfft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Hfft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Hfft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Hfft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Hfftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Hfftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ifft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Ifft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ifft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Ifft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ifftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Ifftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ihfft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Ihfft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ihfft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Ihfft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Ihfftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Ihfftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Irfft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Irfft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Irfft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Irfft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Irfftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Irfftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Rfft, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<int64_t> n, const int64_t dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Rfft (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Rfft2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::vector<int64_t> dim, const std::optional<std::string> norm,
    Tensor out) const {
  throw std::runtime_error(
      "fft::Rfft2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<fft::Rfftn, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> s,
    const std::optional<std::vector<int64_t>> dim,
    const std::optional<std::string> norm, Tensor out) const {
  throw std::runtime_error(
      "fft::Rfftn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fill, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                          const double value,
                                                          Tensor out) const {
  throw std::runtime_error("Fill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fill, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                          const Tensor value,
                                                          Tensor out) const {
  throw std::runtime_error("Fill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fix, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Fix (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloatPower, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor exponent, Tensor out) const {
  throw std::runtime_error(
      "FloatPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloatPower, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double exponent, Tensor out) const {
  throw std::runtime_error(
      "FloatPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloatPower, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double exponent) const {
  throw std::runtime_error(
      "FloatPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloatPower, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor exponent) const {
  throw std::runtime_error(
      "FloatPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Floor, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Floor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloorDivide, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "FloorDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloorDivide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "FloorDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FloorDivide, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "FloorDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmax, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Fmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmin, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Fmin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmod, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const double other,
                                                          Tensor out) const {
  throw std::runtime_error("Fmod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmod, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Fmod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmod, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Fmod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Fmod, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Fmod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Frac, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Frac (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FractionalMaxPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor random_samples,
    const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> output_size, Tensor output,
    Tensor indices) const {
  throw std::runtime_error(
      "FractionalMaxPool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FractionalMaxPool2dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor indices,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> output_size,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "FractionalMaxPool2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FractionalMaxPool3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor random_samples,
    const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> output_size, Tensor output,
    Tensor indices) const {
  throw std::runtime_error(
      "FractionalMaxPool3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FractionalMaxPool3dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor indices,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> output_size,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "FractionalMaxPool3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Frexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor mantissa, Tensor exponent) const {
  throw std::runtime_error("Frexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<FrobeniusNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error(
      "FrobeniusNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gather, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const int64_t dim,
    const bool sparse_grad, Tensor out) const {
  throw std::runtime_error("Gather (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gcd, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Gcd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ge, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Ge (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ge, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Ge (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ge, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Ge (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ge, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Ge (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gelu, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string approximate, Tensor out) const {
  throw std::runtime_error("Gelu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GeluBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const std::string approximate,
    Tensor grad_input) const {
  throw std::runtime_error(
      "GeluBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Geqrf, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor a,
                                                           Tensor tau) const {
  throw std::runtime_error("Geqrf (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ger, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor vec2,
                                                         Tensor out) const {
  throw std::runtime_error("Ger (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Glu, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const int64_t dim,
                                                         Tensor out) const {
  throw std::runtime_error("Glu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GluBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const int64_t dim,
    Tensor grad_input) const {
  throw std::runtime_error(
      "GluBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Greater, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const double other,
                                                             Tensor out) const {
  throw std::runtime_error("Greater (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Greater, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor other,
                                                             Tensor out) const {
  throw std::runtime_error("Greater (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Greater, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Greater (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Greater, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Greater (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GreaterEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "GreaterEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GreaterEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "GreaterEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GreaterEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "GreaterEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<GreaterEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "GreaterEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Gt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Gt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gt, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Gt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Gt, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Gt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hardshrink, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double lambd, Tensor out) const {
  throw std::runtime_error(
      "Hardshrink (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<HardshrinkBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_out, const Tensor input, const double lambd,
    Tensor grad_input) const {
  throw std::runtime_error(
      "HardshrinkBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hardsigmoid, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Hardsigmoid (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<HardsigmoidBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, Tensor grad_input) const {
  throw std::runtime_error(
      "HardsigmoidBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hardswish, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Hardswish (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hardtanh, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double min_val, const double max_val,
    Tensor out) const {
  throw std::runtime_error(
      "Hardtanh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<HardtanhBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const double min_val,
    const double max_val, Tensor grad_input) const {
  throw std::runtime_error(
      "HardtanhBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Heaviside, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor values, Tensor out) const {
  throw std::runtime_error(
      "Heaviside (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Histc, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const int64_t bins,
                                                           const double min,
                                                           const double max,
                                                           Tensor out) const {
  throw std::runtime_error("Histc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Histogram, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor bins, const std::optional<Tensor> weight,
    const bool density, Tensor hist, Tensor bin_edges) const {
  throw std::runtime_error(
      "Histogram (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Histogram, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t bins, const std::optional<Tensor> weight,
    const std::optional<std::vector<double>> range, const bool density,
    Tensor hist, Tensor bin_edges) const {
  throw std::runtime_error(
      "Histogram (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hspmm, Device::Type::kKunlun, 8>::operator()(const Tensor mat1,
                                                           const Tensor mat2,
                                                           Tensor out) const {
  throw std::runtime_error("Hspmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hstack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error("Hstack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<HuberLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const int64_t reduction,
    const double delta, Tensor out) const {
  throw std::runtime_error(
      "HuberLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<HuberLossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const int64_t reduction, const double delta, Tensor grad_input) const {
  throw std::runtime_error(
      "HuberLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Hypot, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor other,
                                                           Tensor out) const {
  throw std::runtime_error("Hypot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<I0, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        Tensor out) const {
  throw std::runtime_error("I0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Igamma, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            const Tensor other,
                                                            Tensor out) const {
  throw std::runtime_error("Igamma (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Igammac, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor other,
                                                             Tensor out) const {
  throw std::runtime_error("Igammac (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Im2col, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> dilation, const std::vector<int64_t> padding,
    const std::vector<int64_t> stride, Tensor out) const {
  throw std::runtime_error("Im2col (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Index, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<std::optional<Tensor>> indices,
    Tensor out) const {
  throw std::runtime_error("Index (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexAdd, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor source,
    const int64_t dim, const double alpha, Tensor out) const {
  throw std::runtime_error(
      "IndexAdd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexCopy, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor source,
    const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "IndexCopy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexFill, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor index, const double value, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error(
      "IndexFill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexFill, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor index, const Tensor value, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error(
      "IndexFill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexReduce, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor source,
    const int64_t dim, const std::string reduce, const bool include_self,
    Tensor out) const {
  throw std::runtime_error(
      "IndexReduce (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<IndexSelect, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error(
      "IndexSelect (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Inner, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor other,
                                                           Tensor out) const {
  throw std::runtime_error("Inner (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::AddRelu, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, const double alpha,
    Tensor out) const {
  throw std::runtime_error(
      "internal::AddRelu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::AddmmActivation, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat1, const Tensor mat2, const double beta,
    const double alpha, const bool use_gelu, Tensor out) const {
  throw std::runtime_error(
      "internal::AddmmActivation (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::BatchNormWithUpdate, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::optional<Tensor> weight,
                             const std::optional<Tensor> bias,
                             Tensor running_mean, Tensor running_var,
                             const double momentum, const double eps,
                             Tensor out, Tensor save_mean, Tensor save_invstd,
                             Tensor reserve) const {
  throw std::runtime_error(
      "internal::BatchNormWithUpdate (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::ChunkCat, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, const int64_t dim,
    const int64_t num_chunks, Tensor out) const {
  throw std::runtime_error(
      "internal::ChunkCat (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::ComputeLinearCombination, Device::Type::kKunlun,
              8>::operator()(const Tensor input, const Tensor coefficients,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::ComputeLinearCombination (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::ConvDepthwise2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, const std::vector<int64_t> dilation,
    Tensor out) const {
  throw std::runtime_error(
      "internal::ConvDepthwise2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::ConvertIndicesFromCooToCsr, Device::Type::kKunlun,
              8>::operator()(const Tensor input, const int64_t size,
                             const bool out_int32, Tensor out) const {
  throw std::runtime_error(
      "internal::ConvertIndicesFromCooToCsr (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::ConvertIndicesFromCsrToCoo, Device::Type::kKunlun,
              8>::operator()(const Tensor crow_indices,
                             const Tensor col_indices, const bool out_int32,
                             const bool transpose, Tensor out) const {
  throw std::runtime_error(
      "internal::ConvertIndicesFromCsrToCoo (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::fft::C2c, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim,
    const int64_t normalization, const bool forward, Tensor out) const {
  throw std::runtime_error(
      "internal::fft::C2c (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::fft::C2r, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim,
    const int64_t normalization, const int64_t last_dim_size,
    Tensor out) const {
  throw std::runtime_error(
      "internal::fft::C2r (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::fft::R2c, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim,
    const int64_t normalization, const bool onesided, Tensor out) const {
  throw std::runtime_error(
      "internal::fft::R2c (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::IntMm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat2, Tensor out) const {
  throw std::runtime_error(
      "internal::IntMm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::linalg::Det, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, Tensor result, Tensor LU, Tensor pivots) const {
  throw std::runtime_error(
      "internal::linalg::Det (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::linalg::Eigh, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const std::string UPLO, const bool compute_v,
    Tensor eigenvalues, Tensor eigenvectors) const {
  throw std::runtime_error(
      "internal::linalg::Eigh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::linalg::Slogdet, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, Tensor sign, Tensor logabsdet, Tensor LU,
    Tensor pivots) const {
  throw std::runtime_error(
      "internal::linalg::Slogdet (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::linalg::SolveEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const Tensor B, const bool left, const bool check_errors,
    Tensor result, Tensor LU, Tensor pivots, Tensor info) const {
  throw std::runtime_error(
      "internal::linalg::SolveEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::linalg::Svd, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool full_matrices, const bool compute_uv,
    const std::optional<std::string> driver, Tensor U, Tensor S,
    Tensor Vh) const {
  throw std::runtime_error(
      "internal::linalg::Svd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::LogSoftmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool half_to_float,
    Tensor out) const {
  throw std::runtime_error(
      "internal::LogSoftmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::LogSoftmaxBackwardData, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor output,
                             const int64_t dim, const DataType input_dtype,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::LogSoftmaxBackwardData (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::Logcumsumexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "internal::Logcumsumexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::NativeBatchNormLegit, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::optional<Tensor> weight,
                             const std::optional<Tensor> bias,
                             Tensor running_mean, Tensor running_var,
                             const bool training, const double momentum,
                             const double eps, Tensor out, Tensor save_mean,
                             Tensor save_invstd) const {
  throw std::runtime_error(
      "internal::NativeBatchNormLegit (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::NativeBatchNormLegit, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::optional<Tensor> weight,
                             const std::optional<Tensor> bias,
                             const bool training, const double momentum,
                             const double eps, Tensor out, Tensor save_mean,
                             Tensor save_invstd) const {
  throw std::runtime_error(
      "internal::NativeBatchNormLegit (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::ScaledMm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat2, const Tensor scale_a,
    const Tensor scale_b, const std::optional<Tensor> bias,
    const std::optional<Tensor> scale_result,
    const std::optional<DataType> out_dtype, const bool use_fast_accum,
    Tensor out) const {
  throw std::runtime_error(
      "internal::ScaledMm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::SlowConv2dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor weight,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> stride,
                             const std::vector<int64_t> padding,
                             Tensor grad_input, Tensor grad_weight,
                             Tensor grad_bias) const {
  throw std::runtime_error(
      "internal::SlowConv2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::SlowConv2dForward, Device::Type::kKunlun,
              8>::operator()(const Tensor input, const Tensor weight,
                             const std::optional<Tensor> bias,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> stride,
                             const std::vector<int64_t> padding,
                             Tensor output) const {
  throw std::runtime_error(
      "internal::SlowConv2dForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::Softmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool half_to_float,
    Tensor out) const {
  throw std::runtime_error(
      "internal::Softmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::SoftmaxBackwardData, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor output,
                             const int64_t dim, const DataType input_dtype,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::SoftmaxBackwardData (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::Stack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "internal::Stack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<internal::UpsampleBicubic2dAa, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::vector<int64_t> output_size,
                             const bool align_corners,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::UpsampleBicubic2dAa (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::UpsampleBicubic2dAaBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const bool align_corners,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::UpsampleBicubic2dAaBackward (ATen backend) is not supported "
      "on Kunlun");
}

template <>
void Operator<internal::UpsampleBilinear2dAa, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::vector<int64_t> output_size,
                             const bool align_corners,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::UpsampleBilinear2dAa (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::UpsampleBilinear2dAaBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const bool align_corners,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::UpsampleBilinear2dAaBackward (ATen backend) is not supported "
      "on Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact1d, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::vector<int64_t> output_size,
                             const std::optional<double> scales,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact1d (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact1dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const std::optional<double> scales,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact1dBackward (ATen backend) is not "
      "supported on Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact2d, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::vector<int64_t> output_size,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact2d (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact2dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact2dBackward (ATen backend) is not "
      "supported on Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact3d, Device::Type::kKunlun,
              8>::operator()(const Tensor input,
                             const std::vector<int64_t> output_size,
                             const std::optional<double> scales_d,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor out) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact3d (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<internal::UpsampleNearestExact3dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const std::optional<double> scales_d,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "internal::UpsampleNearestExact3dBackward (ATen backend) is not "
      "supported on Kunlun");
}

template <>
void Operator<Inverse, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Inverse (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Isin, Device::Type::kKunlun, 8>::operator()(
    const Tensor elements, const Tensor test_elements, const bool assume_unique,
    const bool invert, Tensor out) const {
  throw std::runtime_error("Isin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Isin, Device::Type::kKunlun, 8>::operator()(
    const Tensor elements, const double test_element, const bool assume_unique,
    const bool invert, Tensor out) const {
  throw std::runtime_error("Isin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Isneginf, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Isneginf (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Isposinf, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Isposinf (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Kron, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Kron (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Kthvalue, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t k, const int64_t dim, const bool keepdim,
    Tensor values, Tensor indices) const {
  throw std::runtime_error(
      "Kthvalue (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lcm, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Lcm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ldexp, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor other,
                                                           Tensor out) const {
  throw std::runtime_error("Ldexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Le, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Le (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Le, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Le (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Le, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Le (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Le, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Le (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LeakyRelu, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double negative_slope, Tensor out) const {
  throw std::runtime_error(
      "LeakyRelu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LeakyReluBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const double negative_slope,
    const bool self_is_result, Tensor grad_input) const {
  throw std::runtime_error(
      "LeakyReluBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lerp, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor end,
                                                          const double weight,
                                                          Tensor out) const {
  throw std::runtime_error("Lerp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lerp, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor end,
                                                          const Tensor weight,
                                                          Tensor out) const {
  throw std::runtime_error("Lerp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lerp, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor end, const double weight) const {
  throw std::runtime_error("Lerp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lerp, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor end, const Tensor weight) const {
  throw std::runtime_error("Lerp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Less, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const double other,
                                                          Tensor out) const {
  throw std::runtime_error("Less (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Less, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Less (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Less, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Less (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Less, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Less (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LessEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "LessEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LessEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "LessEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LessEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "LessEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LessEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "LessEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lgamma, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Lgamma (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Cholesky, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool upper, Tensor out) const {
  throw std::runtime_error(
      "linalg::Cholesky (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::CholeskyEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool upper, const bool check_errors, Tensor L,
    Tensor info) const {
  throw std::runtime_error(
      "linalg::CholeskyEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Cond, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> p, Tensor out) const {
  throw std::runtime_error(
      "linalg::Cond (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Cond, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string p, Tensor out) const {
  throw std::runtime_error(
      "linalg::Cond (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Cross, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::Cross (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Det, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, Tensor out) const {
  throw std::runtime_error(
      "linalg::Det (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Eig, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor eigenvalues, Tensor eigenvectors) const {
  throw std::runtime_error(
      "linalg::Eig (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Eigh, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string UPLO, Tensor eigvals,
    Tensor eigvecs) const {
  throw std::runtime_error(
      "linalg::Eigh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Eigvals, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "linalg::Eigvals (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Eigvalsh, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string UPLO, Tensor out) const {
  throw std::runtime_error(
      "linalg::Eigvalsh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::HouseholderProduct, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor tau, Tensor out) const {
  throw std::runtime_error(
      "linalg::HouseholderProduct (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Inv, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, Tensor out) const {
  throw std::runtime_error(
      "linalg::Inv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::InvEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool check_errors, Tensor inverse,
    Tensor info) const {
  throw std::runtime_error(
      "linalg::InvEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LdlFactor, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool hermitian, Tensor LD, Tensor pivots) const {
  throw std::runtime_error(
      "linalg::LdlFactor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LdlFactorEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool hermitian, const bool check_errors,
    Tensor LD, Tensor pivots, Tensor info) const {
  throw std::runtime_error(
      "linalg::LdlFactorEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LdlSolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor LD, const Tensor pivots, const Tensor B, const bool hermitian,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::LdlSolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Lstsq, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor b, const std::optional<double> rcond,
    const std::optional<std::string> driver, Tensor solution, Tensor residuals,
    Tensor rank, Tensor singular_values) const {
  throw std::runtime_error(
      "linalg::Lstsq (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Lu, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool pivot, Tensor P, Tensor L, Tensor U) const {
  throw std::runtime_error(
      "linalg::Lu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LuFactor, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool pivot, Tensor LU, Tensor pivots) const {
  throw std::runtime_error(
      "linalg::LuFactor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LuFactorEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool pivot, const bool check_errors, Tensor LU,
    Tensor pivots, Tensor info) const {
  throw std::runtime_error(
      "linalg::LuFactorEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::LuSolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor LU, const Tensor pivots, const Tensor B, const bool left,
    const bool adjoint, Tensor out) const {
  throw std::runtime_error(
      "linalg::LuSolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Matmul, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "linalg::Matmul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double ord, const std::vector<int64_t> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string ord, const std::vector<int64_t> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixPower, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t n, Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixRank, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double tol, const bool hermitian,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixRank (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixRank, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> atol,
    const std::optional<Tensor> rtol, const bool hermitian, Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixRank (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixRank, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> atol,
    const std::optional<double> rtol, const bool hermitian, Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixRank (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MatrixRank, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor tol, const bool hermitian,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::MatrixRank (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::MultiDot, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error(
      "linalg::MultiDot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Norm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> ord,
    const std::optional<std::vector<int64_t>> dim, const bool keepdim,
    const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error(
      "linalg::Norm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Norm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::string ord,
    const std::optional<std::vector<int64_t>> dim, const bool keepdim,
    const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error(
      "linalg::Norm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Pinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double rcond, const bool hermitian,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::Pinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Pinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> atol,
    const std::optional<Tensor> rtol, const bool hermitian, Tensor out) const {
  throw std::runtime_error(
      "linalg::Pinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Pinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> atol,
    const std::optional<double> rtol, const bool hermitian, Tensor out) const {
  throw std::runtime_error(
      "linalg::Pinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Pinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor rcond, const bool hermitian,
    Tensor out) const {
  throw std::runtime_error(
      "linalg::Pinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Qr, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const std::string mode, Tensor Q, Tensor R) const {
  throw std::runtime_error(
      "linalg::Qr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Slogdet, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, Tensor sign, Tensor logabsdet) const {
  throw std::runtime_error(
      "linalg::Slogdet (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Solve, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const Tensor B, const bool left, Tensor out) const {
  throw std::runtime_error(
      "linalg::Solve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::SolveEx, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const Tensor B, const bool left, const bool check_errors,
    Tensor result, Tensor info) const {
  throw std::runtime_error(
      "linalg::SolveEx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::SolveTriangular, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor B, const bool upper, const bool left,
    const bool unitriangular, Tensor out) const {
  throw std::runtime_error(
      "linalg::SolveTriangular (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Svd, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const bool full_matrices,
    const std::optional<std::string> driver, Tensor U, Tensor S,
    Tensor Vh) const {
  throw std::runtime_error(
      "linalg::Svd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Svdvals, Device::Type::kKunlun, 8>::operator()(
    const Tensor A, const std::optional<std::string> driver, Tensor out) const {
  throw std::runtime_error(
      "linalg::Svdvals (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Tensorinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t ind, Tensor out) const {
  throw std::runtime_error(
      "linalg::Tensorinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Tensorsolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other,
    const std::optional<std::vector<int64_t>> dims, Tensor out) const {
  throw std::runtime_error(
      "linalg::Tensorsolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::Vecdot, Device::Type::kKunlun, 8>::operator()(
    const Tensor x, const Tensor y, const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "linalg::Vecdot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<linalg::VectorNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double ord,
    const std::optional<std::vector<int64_t>> dim, const bool keepdim,
    const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error(
      "linalg::VectorNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Linear, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    Tensor out) const {
  throw std::runtime_error("Linear (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Linspace, Device::Type::kKunlun, 8>::operator()(
    const Tensor start, const Tensor end, const int64_t steps,
    Tensor out) const {
  throw std::runtime_error(
      "Linspace (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Linspace, Device::Type::kKunlun, 8>::operator()(
    const Tensor start, const double end, const int64_t steps,
    Tensor out) const {
  throw std::runtime_error(
      "Linspace (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Log, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Log (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Log10, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Log10 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Log1p, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Log1p (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Log2, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Log2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogSigmoid, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "LogSigmoid (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogSigmoidBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor buffer,
    Tensor grad_input) const {
  throw std::runtime_error(
      "LogSigmoidBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogSigmoidForward, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor output, Tensor buffer) const {
  throw std::runtime_error(
      "LogSigmoidForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogSoftmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const std::optional<DataType> dtype,
    Tensor out) const {
  throw std::runtime_error(
      "LogSoftmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logaddexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Logaddexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logaddexp2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Logaddexp2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logcumsumexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, Tensor out) const {
  throw std::runtime_error(
      "Logcumsumexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogicalAnd, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "LogicalAnd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogicalNot, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "LogicalNot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogicalOr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "LogicalOr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogicalXor, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "LogicalXor (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logit, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> eps, Tensor out) const {
  throw std::runtime_error("Logit (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LogitBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::optional<double> eps, Tensor grad_input) const {
  throw std::runtime_error(
      "LogitBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logspace, Device::Type::kKunlun, 8>::operator()(
    const Tensor start, const Tensor end, const int64_t steps,
    const double base, Tensor out) const {
  throw std::runtime_error(
      "Logspace (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logspace, Device::Type::kKunlun, 8>::operator()(
    const Tensor start, const double end, const int64_t steps,
    const double base, Tensor out) const {
  throw std::runtime_error(
      "Logspace (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Logsumexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error(
      "Logsumexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Lt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Lt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lt, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Lt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Lt, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Lt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LuSolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor LU_data, const Tensor LU_pivots,
    Tensor out) const {
  throw std::runtime_error("LuSolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<LuUnpack, Device::Type::kKunlun, 8>::operator()(
    const Tensor LU_data, const Tensor LU_pivots, const bool unpack_data,
    const bool unpack_pivots, Tensor P, Tensor L, Tensor U) const {
  throw std::runtime_error(
      "LuUnpack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaskedFill, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor mask, const double value, Tensor out) const {
  throw std::runtime_error(
      "MaskedFill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaskedFill, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor mask, const Tensor value, Tensor out) const {
  throw std::runtime_error(
      "MaskedFill (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaskedSelect, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mask, Tensor out) const {
  throw std::runtime_error(
      "MaskedSelect (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Matmul, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            const Tensor other,
                                                            Tensor out) const {
  throw std::runtime_error("Matmul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MatrixPower, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t n, Tensor out) const {
  throw std::runtime_error(
      "MatrixPower (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Max, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Max (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Max, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim, Tensor max,
    Tensor max_values) const {
  throw std::runtime_error("Max (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Max, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Max (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxPool2dWithIndices, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const std::vector<int64_t> dilation, const bool ceil_mode, Tensor out,
    Tensor indices) const {
  throw std::runtime_error(
      "MaxPool2dWithIndices (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxPool2dWithIndicesBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor indices,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> stride,
                             const std::vector<int64_t> padding,
                             const std::vector<int64_t> dilation,
                             const bool ceil_mode, Tensor grad_input) const {
  throw std::runtime_error(
      "MaxPool2dWithIndicesBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxPool3dWithIndices, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const std::vector<int64_t> dilation, const bool ceil_mode, Tensor out,
    Tensor indices) const {
  throw std::runtime_error(
      "MaxPool3dWithIndices (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxPool3dWithIndicesBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor indices,
                             const std::vector<int64_t> kernel_size,
                             const std::vector<int64_t> stride,
                             const std::vector<int64_t> padding,
                             const std::vector<int64_t> dilation,
                             const bool ceil_mode, Tensor grad_input) const {
  throw std::runtime_error(
      "MaxPool3dWithIndicesBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxUnpool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor indices,
    const std::vector<int64_t> output_size, Tensor out) const {
  throw std::runtime_error(
      "MaxUnpool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxUnpool3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor indices,
    const std::vector<int64_t> output_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "MaxUnpool3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Maximum, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor other,
                                                             Tensor out) const {
  throw std::runtime_error("Maximum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mean, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Mean (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mean, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Mean (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Median, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim, Tensor values,
    Tensor indices) const {
  throw std::runtime_error("Median (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Min, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Min (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Min, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim, Tensor min,
    Tensor min_indices) const {
  throw std::runtime_error("Min (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Min, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Min (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Minimum, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor other,
                                                             Tensor out) const {
  throw std::runtime_error("Minimum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mish, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Mish (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MkldnnAdaptiveAvgPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    Tensor out) const {
  throw std::runtime_error(
      "MkldnnAdaptiveAvgPool2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mm, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor mat2,
                                                        Tensor out) const {
  throw std::runtime_error("Mm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mode, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim, Tensor values,
    Tensor indices) const {
  throw std::runtime_error("Mode (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MseLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const int64_t reduction,
    Tensor out) const {
  throw std::runtime_error("MseLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MseLossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const int64_t reduction, Tensor grad_input) const {
  throw std::runtime_error(
      "MseLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Msort, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Msort (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mul, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                         const Tensor other,
                                                         Tensor out) const {
  throw std::runtime_error("Mul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mul, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                         const double other,
                                                         Tensor out) const {
  throw std::runtime_error("Mul (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MultiMarginLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const double p,
    const double margin, const std::optional<Tensor> weight,
    const int64_t reduction, Tensor out) const {
  throw std::runtime_error(
      "MultiMarginLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MultiMarginLossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const std::optional<Tensor> weight, const double p, const double margin,
    const int64_t reduction, Tensor grad_input) const {
  throw std::runtime_error(
      "MultiMarginLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MultilabelMarginLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const int64_t reduction,
    Tensor out) const {
  throw std::runtime_error(
      "MultilabelMarginLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MultilabelMarginLossBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output, const Tensor input,
                             const Tensor target, const Tensor is_target,
                             const int64_t reduction, Tensor grad_input) const {
  throw std::runtime_error(
      "MultilabelMarginLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MultilabelMarginLossForward, Device::Type::kKunlun,
              8>::operator()(const Tensor input, const Tensor target,
                             const int64_t reduction, Tensor output,
                             Tensor is_target) const {
  throw std::runtime_error(
      "MultilabelMarginLossForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Multinomial, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t num_samples, const bool replacement,
    Tensor out) const {
  throw std::runtime_error(
      "Multinomial (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Multiply, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Multiply (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Multiply, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "Multiply (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Multiply, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "Multiply (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mv, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor vec,
                                                        Tensor out) const {
  throw std::runtime_error("Mv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Mvlgamma, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t p, Tensor out) const {
  throw std::runtime_error(
      "Mvlgamma (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NanToNum, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> nan,
    const std::optional<double> posinf, const std::optional<double> neginf,
    Tensor out) const {
  throw std::runtime_error(
      "NanToNum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nanmean, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Nanmean (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nanmedian, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim, Tensor values,
    Tensor indices) const {
  throw std::runtime_error(
      "Nanmedian (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nanquantile, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor q, const std::optional<int64_t> dim,
    const bool keepdim, const std::string interpolation, Tensor out) const {
  throw std::runtime_error(
      "Nanquantile (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nanquantile, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double q, const std::optional<int64_t> dim,
    const bool keepdim, const std::string interpolation, Tensor out) const {
  throw std::runtime_error(
      "Nanquantile (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nansum, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Nansum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NarrowCopy, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const int64_t start,
    const int64_t length, Tensor out) const {
  throw std::runtime_error(
      "NarrowCopy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NativeBatchNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<Tensor> weight,
    const std::optional<Tensor> bias, const std::optional<Tensor> running_mean,
    const std::optional<Tensor> running_var, const bool training,
    const double momentum, const double eps, Tensor out, Tensor save_mean,
    Tensor save_invstd) const {
  throw std::runtime_error(
      "NativeBatchNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NativeLayerNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> normalized_shape,
    const std::optional<Tensor> weight, const std::optional<Tensor> bias,
    const double eps, Tensor out0, Tensor out1, Tensor out2) const {
  throw std::runtime_error(
      "NativeLayerNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ne, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const double other,
                                                        Tensor out) const {
  throw std::runtime_error("Ne (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ne, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const Tensor other,
                                                        Tensor out) const {
  throw std::runtime_error("Ne (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ne, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Ne (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ne, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Ne (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Neg, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Neg (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Negative, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Negative (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nextafter, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Nextafter (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const std::optional<Tensor> weight,
    const int64_t reduction, const int64_t ignore_index, Tensor out) const {
  throw std::runtime_error("NllLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLoss2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const std::optional<Tensor> weight,
    const int64_t reduction, const int64_t ignore_index, Tensor out) const {
  throw std::runtime_error(
      "NllLoss2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLoss2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const std::optional<Tensor> weight, const Tensor total_weight,
    const int64_t reduction, const int64_t ignore_index,
    Tensor grad_input) const {
  throw std::runtime_error(
      "NllLoss2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLoss2dForward, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const std::optional<Tensor> weight,
    const int64_t reduction, const int64_t ignore_index, Tensor output,
    Tensor total_weight) const {
  throw std::runtime_error(
      "NllLoss2dForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const std::optional<Tensor> weight, const Tensor total_weight,
    const int64_t reduction, const int64_t ignore_index,
    Tensor grad_input) const {
  throw std::runtime_error(
      "NllLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NllLossForward, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const std::optional<Tensor> weight,
    const int64_t reduction, const int64_t ignore_index, Tensor output,
    Tensor total_weight) const {
  throw std::runtime_error(
      "NllLossForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Nonzero, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Nonzero (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NonzeroStatic, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t size, const int64_t fill_value,
    Tensor out) const {
  throw std::runtime_error(
      "NonzeroStatic (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Norm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> p,
    const std::vector<int64_t> dim, const bool keepdim, Tensor out) const {
  throw std::runtime_error("Norm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Normal, Device::Type::kKunlun, 8>::operator()(const Tensor mean,
                                                            const double std,
                                                            Tensor out) const {
  throw std::runtime_error("Normal (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Normal, Device::Type::kKunlun, 8>::operator()(const Tensor mean,
                                                            const Tensor std,
                                                            Tensor out) const {
  throw std::runtime_error("Normal (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NotEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "NotEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NotEqual, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "NotEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NotEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "NotEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NotEqual, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "NotEqual (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NuclearNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const bool keepdim, Tensor out) const {
  throw std::runtime_error(
      "NuclearNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<NuclearNorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error(
      "NuclearNorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Orgqr, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor input2,
                                                           Tensor out) const {
  throw std::runtime_error("Orgqr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Ormqr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor input2, const Tensor input3,
    const bool left, const bool transpose, Tensor out) const {
  throw std::runtime_error("Ormqr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Outer, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor vec2,
                                                           Tensor out) const {
  throw std::runtime_error("Outer (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Polar, Device::Type::kKunlun, 8>::operator()(const Tensor abs,
                                                           const Tensor angle,
                                                           Tensor out) const {
  throw std::runtime_error("Polar (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Pow, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor exponent,
                                                         Tensor out) const {
  throw std::runtime_error("Pow (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Pow, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const double exponent,
                                                         Tensor out) const {
  throw std::runtime_error("Pow (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Pow, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double exponent) const {
  throw std::runtime_error("Pow (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Pow, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor exponent) const {
  throw std::runtime_error("Pow (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Prod, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool keepdim,
    const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Prod (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Qr, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                        const bool some,
                                                        Tensor Q,
                                                        Tensor R) const {
  throw std::runtime_error("Qr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Quantile, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor q, const std::optional<int64_t> dim,
    const bool keepdim, const std::string interpolation, Tensor out) const {
  throw std::runtime_error(
      "Quantile (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Quantile, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double q, const std::optional<int64_t> dim,
    const bool keepdim, const std::string interpolation, Tensor out) const {
  throw std::runtime_error(
      "Quantile (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Rad2deg, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Rad2deg (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Random, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const int64_t from, const std::optional<int64_t> to,
    Tensor out) const {
  throw std::runtime_error("Random (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Random, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                            const int64_t to,
                                                            Tensor out) const {
  throw std::runtime_error("Random (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Reciprocal, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "Reciprocal (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad1d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReflectionPad1d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad1dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReflectionPad1dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReflectionPad2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReflectionPad2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReflectionPad3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReflectionPad3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReflectionPad3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Relu, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Relu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Remainder, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "Remainder (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Remainder, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "Remainder (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Remainder, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error(
      "Remainder (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Remainder, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "Remainder (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Renorm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double p, const int64_t dim, const double maxnorm,
    Tensor out) const {
  throw std::runtime_error("Renorm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad1d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReplicationPad1d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad1dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReplicationPad1dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReplicationPad2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReplicationPad2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ReplicationPad3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ReplicationPad3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input,
    const std::vector<int64_t> padding, Tensor grad_input) const {
  throw std::runtime_error(
      "ReplicationPad3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Round, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Round (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Round, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t decimals, Tensor out) const {
  throw std::runtime_error("Round (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<RowStack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error(
      "RowStack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<RreluWithNoise, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double lower, const double upper,
    const bool training, Tensor out, const Tensor noise) const {
  throw std::runtime_error(
      "RreluWithNoise (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Rsqrt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Rsqrt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Scatter, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor index,
                                                             const Tensor src,
                                                             const int64_t dim,
                                                             Tensor out) const {
  throw std::runtime_error("Scatter (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Scatter, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             const Tensor index,
                                                             const double value,
                                                             const int64_t dim,
                                                             Tensor out) const {
  throw std::runtime_error("Scatter (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Scatter, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor src, const int64_t dim,
    const std::string reduce, Tensor out) const {
  throw std::runtime_error("Scatter (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Scatter, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const double value,
    const int64_t dim, const std::string reduce, Tensor out) const {
  throw std::runtime_error("Scatter (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ScatterAdd, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor src, const int64_t dim,
    Tensor out) const {
  throw std::runtime_error(
      "ScatterAdd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ScatterReduce, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor index, const Tensor src, const int64_t dim,
    const std::string reduce, const bool include_self, Tensor out) const {
  throw std::runtime_error(
      "ScatterReduce (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Searchsorted, Device::Type::kKunlun, 8>::operator()(
    const Tensor sorted_sequence, const Tensor input, const bool out_int32,
    const bool right, const std::optional<std::string> side,
    const std::optional<Tensor> sorter, Tensor out) const {
  throw std::runtime_error(
      "Searchsorted (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Searchsorted, Device::Type::kKunlun, 8>::operator()(
    const Tensor sorted_sequence, const double input, const bool out_int32,
    const bool right, const std::optional<std::string> side,
    const std::optional<Tensor> sorter, Tensor out) const {
  throw std::runtime_error(
      "Searchsorted (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Set, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor source, const int64_t storage_offset,
    const std::vector<int64_t> size, const std::vector<int64_t> stride,
    Tensor out) const {
  throw std::runtime_error("Set (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Set, Device::Type::kKunlun, 8>::operator()(Tensor input,
                                                         const Tensor source,
                                                         Tensor out) const {
  throw std::runtime_error("Set (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sgn, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Sgn (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sigmoid, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Sigmoid (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SigmoidBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor output, Tensor grad_input) const {
  throw std::runtime_error(
      "SigmoidBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sign, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Sign (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Signbit, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                             Tensor out) const {
  throw std::runtime_error("Signbit (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Silu, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Silu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SiluBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, Tensor grad_input) const {
  throw std::runtime_error(
      "SiluBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sin, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Sin (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sinc, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Sinc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sinh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Sinh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Slogdet, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor sign, Tensor logabsdet) const {
  throw std::runtime_error("Slogdet (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SlowConv3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "SlowConv3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SlowConv3dForward, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, Tensor output) const {
  throw std::runtime_error(
      "SlowConv3dForward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SlowConvTranspose2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding,
    const std::vector<int64_t> output_padding,
    const std::vector<int64_t> dilation, Tensor out) const {
  throw std::runtime_error(
      "SlowConvTranspose2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SlowConvTranspose3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding,
    const std::vector<int64_t> output_padding,
    const std::vector<int64_t> dilation, Tensor out) const {
  throw std::runtime_error(
      "SlowConvTranspose3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SmoothL1Loss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const int64_t reduction,
    const double beta, Tensor out) const {
  throw std::runtime_error(
      "SmoothL1Loss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SmoothL1LossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const int64_t reduction, const double beta, Tensor grad_input) const {
  throw std::runtime_error(
      "SmoothL1LossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SoftMarginLoss, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor target, const int64_t reduction,
    Tensor out) const {
  throw std::runtime_error(
      "SoftMarginLoss (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SoftMarginLossBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const Tensor target,
    const int64_t reduction, Tensor grad_input) const {
  throw std::runtime_error(
      "SoftMarginLossBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Softmax, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const std::optional<DataType> dtype,
    Tensor out) const {
  throw std::runtime_error("Softmax (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Softplus, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double beta, const double threshold,
    Tensor out) const {
  throw std::runtime_error(
      "Softplus (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SoftplusBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const double beta,
    const double threshold, Tensor grad_input) const {
  throw std::runtime_error(
      "SoftplusBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Softshrink, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double lambd, Tensor out) const {
  throw std::runtime_error(
      "Softshrink (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SoftshrinkBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const double lambd,
    Tensor grad_input) const {
  throw std::runtime_error(
      "SoftshrinkBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sort, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t dim, const bool descending, Tensor values,
    Tensor indices) const {
  throw std::runtime_error("Sort (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sort, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<bool> stable, const int64_t dim,
    const bool descending, Tensor values, Tensor indices) const {
  throw std::runtime_error("Sort (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<SparseSampledAddmm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat1, const Tensor mat2, const double beta,
    const double alpha, Tensor out) const {
  throw std::runtime_error(
      "SparseSampledAddmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::AiryAi, Device::Type::kKunlun, 8>::operator()(
    const Tensor x, Tensor out) const {
  throw std::runtime_error(
      "special::AiryAi (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::BesselJ0, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::BesselJ0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::BesselJ1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::BesselJ1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::BesselY0, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::BesselY0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::BesselY1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::BesselY1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialT, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialT (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialT, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialT (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialU, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialU (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialU, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialU (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialV, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialV (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialV, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialV (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialW, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialW (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ChebyshevPolynomialW, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ChebyshevPolynomialW (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::Digamma, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Digamma (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Entr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Entr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Erf, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Erf (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Erfc, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Erfc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Erfcx, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Erfcx (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Erfinv, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Erfinv (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Exp2, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Exp2 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Expit, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Expit (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Expm1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Expm1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Gammainc, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "special::Gammainc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Gammaincc, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "special::Gammaincc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Gammaln, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Gammaln (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::HermitePolynomialH, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::HermitePolynomialH (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::HermitePolynomialH, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::HermitePolynomialH (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::HermitePolynomialHe, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::HermitePolynomialHe (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::HermitePolynomialHe, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::HermitePolynomialHe (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::I0, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::I0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::I0e, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::I0e (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::I1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::I1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::I1e, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::I1e (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::LaguerrePolynomialL, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::LaguerrePolynomialL (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::LaguerrePolynomialL, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::LaguerrePolynomialL (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::LegendrePolynomialP, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::LegendrePolynomialP (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::LegendrePolynomialP, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::LegendrePolynomialP (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Log1p, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Log1p (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::LogNdtr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::LogNdtr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Logit, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<double> eps, Tensor out) const {
  throw std::runtime_error(
      "special::Logit (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Logsumexp, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error(
      "special::Logsumexp (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ModifiedBesselI0, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::ModifiedBesselI0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ModifiedBesselI1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::ModifiedBesselI1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ModifiedBesselK0, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::ModifiedBesselK0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ModifiedBesselK1, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::ModifiedBesselK1 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Multigammaln, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t p, Tensor out) const {
  throw std::runtime_error(
      "special::Multigammaln (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Ndtr, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Ndtr (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Ndtri, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Ndtri (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Psi, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Psi (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Round, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t decimals, Tensor out) const {
  throw std::runtime_error(
      "special::Round (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::ScaledModifiedBesselK0, Device::Type::kKunlun,
              8>::operator()(const Tensor x, Tensor out) const {
  throw std::runtime_error(
      "special::ScaledModifiedBesselK0 (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ScaledModifiedBesselK1, Device::Type::kKunlun,
              8>::operator()(const Tensor x, Tensor out) const {
  throw std::runtime_error(
      "special::ScaledModifiedBesselK1 (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialT, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialT (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialT, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialT (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialU, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialU (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialU, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialU (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialV, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialV (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialV, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialV (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialW, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const Tensor n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialW (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::ShiftedChebyshevPolynomialW, Device::Type::kKunlun,
              8>::operator()(const Tensor x, const double n, Tensor out) const {
  throw std::runtime_error(
      "special::ShiftedChebyshevPolynomialW (ATen backend) is not supported on "
      "Kunlun");
}

template <>
void Operator<special::Sinc, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, Tensor out) const {
  throw std::runtime_error(
      "special::Sinc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::SphericalBesselJ0, Device::Type::kKunlun, 8>::operator()(
    const Tensor x, Tensor out) const {
  throw std::runtime_error(
      "special::SphericalBesselJ0 (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Xlog1py, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "special::Xlog1py (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Xlog1py, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "special::Xlog1py (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Xlogy, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "special::Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Xlogy, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "special::Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Zeta, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "special::Zeta (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<special::Zeta, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "special::Zeta (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sqrt, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Sqrt (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Square, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                            Tensor out) const {
  throw std::runtime_error("Square (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sspaddmm, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor mat1, const Tensor mat2, const double beta,
    const double alpha, Tensor out) const {
  throw std::runtime_error(
      "Sspaddmm (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Stack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, const int64_t dim, Tensor out) const {
  throw std::runtime_error("Stack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Std, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool unbiased, const bool keepdim, Tensor out) const {
  throw std::runtime_error("Std (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Std, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const std::optional<double> correction, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Std (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sub, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const Tensor other,
                                                         const double alpha,
                                                         Tensor out) const {
  throw std::runtime_error("Sub (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sub, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other, const double alpha) const {
  throw std::runtime_error("Sub (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sub, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const double other,
                                                         const double alpha,
                                                         Tensor out) const {
  throw std::runtime_error("Sub (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Subtract, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, const double alpha,
    Tensor out) const {
  throw std::runtime_error(
      "Subtract (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Subtract, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other, const double alpha) const {
  throw std::runtime_error(
      "Subtract (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Subtract, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other, const double alpha, Tensor out) const {
  throw std::runtime_error(
      "Subtract (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Sum, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool keepdim, const std::optional<DataType> dtype, Tensor out) const {
  throw std::runtime_error("Sum (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Svd, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         const bool some,
                                                         const bool compute_uv,
                                                         Tensor U, Tensor S,
                                                         Tensor V) const {
  throw std::runtime_error("Svd (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Take, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor index,
                                                          Tensor out) const {
  throw std::runtime_error("Take (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TakeAlongDim, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor indices, const std::optional<int64_t> dim,
    Tensor out) const {
  throw std::runtime_error(
      "TakeAlongDim (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Tan, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                         Tensor out) const {
  throw std::runtime_error("Tan (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Tanh, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          Tensor out) const {
  throw std::runtime_error("Tanh (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TanhBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor output, Tensor grad_input) const {
  throw std::runtime_error(
      "TanhBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Tensordot, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other,
    const std::vector<int64_t> dims_self, const std::vector<int64_t> dims_other,
    Tensor out) const {
  throw std::runtime_error(
      "Tensordot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ThnnConv2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor weight, const std::optional<Tensor> bias,
    const std::vector<int64_t> kernel_size, const std::vector<int64_t> stride,
    const std::vector<int64_t> padding, Tensor out) const {
  throw std::runtime_error(
      "ThnnConv2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Threshold, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const double threshold, const double value,
    Tensor out) const {
  throw std::runtime_error(
      "Threshold (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<ThresholdBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const Tensor input, const double threshold,
    Tensor grad_input) const {
  throw std::runtime_error(
      "ThresholdBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Topk, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t k, const int64_t dim, const bool largest,
    const bool sorted, Tensor values, Tensor indices) const {
  throw std::runtime_error("Topk (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TriangularSolve, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor A, const bool upper, const bool transpose,
    const bool unitriangular, Tensor X, Tensor M) const {
  throw std::runtime_error(
      "TriangularSolve (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Tril, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t diagonal, Tensor out) const {
  throw std::runtime_error("Tril (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Triu, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const int64_t diagonal, Tensor out) const {
  throw std::runtime_error("Triu (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TrueDivide, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const Tensor other, Tensor out) const {
  throw std::runtime_error(
      "TrueDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TrueDivide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error(
      "TrueDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<TrueDivide, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other, Tensor out) const {
  throw std::runtime_error(
      "TrueDivide (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Trunc, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           Tensor out) const {
  throw std::runtime_error("Trunc (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleBicubic2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const bool align_corners, const std::optional<double> scales_h,
    const std::optional<double> scales_w, Tensor out) const {
  throw std::runtime_error(
      "UpsampleBicubic2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleBicubic2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const bool align_corners,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleBicubic2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleBilinear2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const bool align_corners, const std::optional<double> scales_h,
    const std::optional<double> scales_w, Tensor out) const {
  throw std::runtime_error(
      "UpsampleBilinear2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleBilinear2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const bool align_corners,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleBilinear2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleLinear1d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const bool align_corners, const std::optional<double> scales,
    Tensor out) const {
  throw std::runtime_error(
      "UpsampleLinear1d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleLinear1dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const bool align_corners,
    const std::optional<double> scales, Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleLinear1dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest1d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const std::optional<double> scales, Tensor out) const {
  throw std::runtime_error(
      "UpsampleNearest1d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest1dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const std::optional<double> scales,
    Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleNearest1dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    Tensor out) const {
  throw std::runtime_error(
      "UpsampleNearest2d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest2dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const std::optional<double> scales_h,
    const std::optional<double> scales_w, Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleNearest2dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const std::optional<double> scales_d, const std::optional<double> scales_h,
    const std::optional<double> scales_w, Tensor out) const {
  throw std::runtime_error(
      "UpsampleNearest3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleNearest3dBackward, Device::Type::kKunlun, 8>::operator()(
    const Tensor grad_output, const std::vector<int64_t> output_size,
    const std::vector<int64_t> input_size, const std::optional<double> scales_d,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleNearest3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleTrilinear3d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> output_size,
    const bool align_corners, const std::optional<double> scales_d,
    const std::optional<double> scales_h, const std::optional<double> scales_w,
    Tensor out) const {
  throw std::runtime_error(
      "UpsampleTrilinear3d (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<UpsampleTrilinear3dBackward, Device::Type::kKunlun,
              8>::operator()(const Tensor grad_output,
                             const std::vector<int64_t> output_size,
                             const std::vector<int64_t> input_size,
                             const bool align_corners,
                             const std::optional<double> scales_d,
                             const std::optional<double> scales_h,
                             const std::optional<double> scales_w,
                             Tensor grad_input) const {
  throw std::runtime_error(
      "UpsampleTrilinear3dBackward (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Var, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const bool unbiased, const bool keepdim, Tensor out) const {
  throw std::runtime_error("Var (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Var, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::optional<std::vector<int64_t>> dim,
    const std::optional<double> correction, const bool keepdim,
    Tensor out) const {
  throw std::runtime_error("Var (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Vdot, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                          const Tensor other,
                                                          Tensor out) const {
  throw std::runtime_error("Vdot (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Vstack, Device::Type::kKunlun, 8>::operator()(
    const std::vector<Tensor> tensors, Tensor out) const {
  throw std::runtime_error("Vstack (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Where, Device::Type::kKunlun, 8>::operator()(
    const Tensor condition, const Tensor input, const Tensor other,
    Tensor out) const {
  throw std::runtime_error("Where (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Xlogy, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const Tensor other,
                                                           Tensor out) const {
  throw std::runtime_error("Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Xlogy, Device::Type::kKunlun, 8>::operator()(const Tensor input,
                                                           const double other,
                                                           Tensor out) const {
  throw std::runtime_error("Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Xlogy, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const Tensor other) const {
  throw std::runtime_error("Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<Xlogy, Device::Type::kKunlun, 8>::operator()(
    Tensor input, const double other) const {
  throw std::runtime_error("Xlogy (ATen backend) is not supported on Kunlun");
}

template <>
void Operator<MaxPool2d, Device::Type::kKunlun, 8>::operator()(
    const Tensor input, const std::vector<int64_t> kernel_size,
    const std::vector<int64_t> stride, const std::vector<int64_t> padding,
    const std::vector<int64_t> dilation, bool ceil_mode, Tensor out) const {
  throw std::runtime_error(
      "MaxPool2d (ATen backend) is not supported on Kunlun");
}

}  // namespace infini::ops
