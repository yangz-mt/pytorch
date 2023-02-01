import functools
import logging
import math
import numbers

import torch
import torch._decomp as decomp
from torch import Tensor
from torch._decomp import get_decompositions
from torch._prims_common import is_boolean_dtype, is_integer_dtype
from torch.utils._mode_utils import no_dispatch

from . import config, utils

log = logging.getLogger(__name__)
aten = torch.ops.aten
log = logging.getLogger(__name__)

decompositions = get_decompositions(
    [
        aten.linspace,
        aten.logaddexp,
        aten._adaptive_avg_pool2d_backward,
        aten.addcmul,
        aten.addcmul_,
        aten.addcdiv_,
        aten.arange,
        aten.avg_pool2d_backward,
        aten.binary_cross_entropy_with_logits,
        aten.clamp_max,
        aten.clamp_min,
        aten.col2im,
        aten.cudnn_batch_norm,
        aten.cudnn_batch_norm_backward,
        aten.detach,
        aten.dot,
        aten.elu,
        aten.elu_backward,
        aten._embedding_bag,
        aten.embedding_dense_backward,
        aten.expand_as,
        aten.eye,
        aten.ones_like,
        aten.zeros_like,
        aten.zeros,
        aten.ones,
        aten.fill,
        aten.flip,
        aten._fused_moving_avg_obs_fq_helper,
        aten.gelu,
        aten.gelu_backward,
        aten.glu_backward,
        aten.grid_sampler_2d,
        aten.hardsigmoid,
        aten.hardsigmoid_backward,
        aten.upsample_bilinear2d,
        aten.hardswish,
        aten.hardswish_,
        aten.hardswish_backward,
        aten.hardtanh,
        aten.hardtanh_,
        aten.hardtanh_backward,
        aten.im2col,
        aten.index_select,
        aten.index_add,
        aten.index_add_,
        aten.index_copy,
        aten.index_copy_,
        aten.index_fill,
        aten.index_fill_,
        aten.l1_loss,
        aten.leaky_relu,
        aten.leaky_relu_,
        aten.leaky_relu_backward,
        aten.linalg_vector_norm,
        aten.logit,
        aten.logit_backward,
        aten._log_softmax,
        aten._log_softmax_backward_data,
        aten.logsumexp.default,
        aten.masked_fill,
        aten.masked_fill_,
        aten.max_pool2d_with_indices_backward,
        aten.mse_loss,
        aten.mse_loss_backward,
        aten.mv,
        aten.narrow,
        aten.native_batch_norm,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten.native_batch_norm_backward,
        aten.native_dropout_backward,
        aten.native_group_norm,
        aten.native_group_norm_backward,
        aten.native_layer_norm,
        aten.native_layer_norm_backward,
        aten.new_empty,
        aten.new_full,
        aten.new_zeros,
        aten.new_ones,
        aten.nll_loss_backward,
        aten.nll_loss_forward,
        aten.norm,
        aten._reshape_alias,
        aten.select_backward,
        aten.select_scatter,
        aten.sgn,
        aten.sigmoid_backward,
        aten.silu,
        aten.silu_,
        aten.silu_backward,
        aten.slice_backward,
        aten._softmax,
        aten._softmax_backward_data,
        aten.softplus,
        aten.softplus_backward,
        aten.stack,
        aten.std_mean.correction,
        aten.t,
        aten.tanh_backward,
        aten.threshold_backward,
        aten._to_copy,
        aten.transpose.int,
        aten.tril.default,
        aten.unfold,
        aten.unfold_backward,
        aten.upsample_bilinear2d.vec,
        aten.upsample_nearest2d_backward,
        aten.bucketize,
        aten.zero_,
        aten.zero,
    ]
)


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning(f"duplicate decomp: {ops}")
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition([aten.clamp])
def clamp(x, min=None, max=None):
    if min is not None:
        x = torch.maximum(x, torch.tensor(min, dtype=x.dtype, device=x.device))
    if max is not None:
        x = torch.minimum(x, torch.tensor(max, dtype=x.dtype, device=x.device))
    return x


# TorchInductor-only decomposition. It should not be taken to core.
# See https://github.com/pytorch/torchdynamo/pull/1120
@register_decomposition([aten.floor_divide.default])
def floordiv(a, b):
    return aten.div.Tensor_mode(a, b, rounding_mode="floor")


def get_alignment_size(x):
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor):
    return a.is_cuda and b.is_cuda


def get_padded_length(x, alignment_size):
    if alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x, padded_length, dim):
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


@register_decomposition([aten.addmm])
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.addmm, input=input)
    ):
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
        if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
            return pad_addmm(
                input, mat1, mat2, m_padded_length, n_padded_length, k_padded_length
            )

    return NotImplemented  # go directly to lowering


def pad_addmm(input, mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # addmm decomp with padding will go through pad_addmm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
    elif m_padded_length != 0:
        mat1 = pad_dim(mat1, m_padded_length, 0)

    if input is not None and k_padded_length == 0:
        if n_padded_length != 0:
            if input.dim() == 2:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1:
                input = pad_dim(input, n_padded_length, 0)
        elif m_padded_length != 0 and input.dim() == 2:
            input = pad_dim(input, m_padded_length, 0)

    if k_padded_length != 0:
        return torch.ops.aten.addmm(input, mat1, mat2)
    elif n_padded_length != 0:
        return torch.ops.aten.addmm(input, mat1, mat2)[:, :-n_padded_length]
    else:
        return torch.ops.aten.addmm(input, mat1, mat2)[:-m_padded_length, :]


def should_pad_bench(mat1, mat2, op, input=None):
    assert utils.has_triton()
    from triton.testing import do_bench

    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
            k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
            n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
        elif op is torch.ops.aten.bmm:
            m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
            k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
            n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
        else:
            return False

        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

        mat1 = torch.randn_like(mat1)
        mat2 = torch.randn_like(mat2)
        warmup = 5
        rep = 100
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(
                lambda: op(mat1, mat2), warmup=warmup, rep=rep, fast_flush=True
            )[0]
        else:
            if input is not None:
                input = torch.randn_like(input)
            ori_time = do_bench(
                lambda: op(input, mat1, mat2), warmup=warmup, rep=rep, fast_flush=True
            )[0]

        mat1_pad = torch.randn_like(mat1)
        mat2_pad = torch.randn_like(mat2)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
                warmup=warmup,
                rep=rep,
                fast_flush=True,
            )[0]
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
                warmup=warmup,
                rep=rep,
                fast_flush=True,
            )[0]
        else:
            pad_time = do_bench(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
                warmup=warmup,
                rep=rep,
                fast_flush=True,
            )[0]

        # Shape padding introduces addtional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
        # tradeoff between performance improvement from shape padding and overhead from addtional memory ops
        # TODO: Build a learned model which would be better than this heuristic
        return ori_time > pad_time * 1.1


@register_decomposition([aten.mm])
def mm_decomp(mat1, mat2):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.mm)
    ):
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))

        if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
            return pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)

    return NotImplemented  # go directly to lowering


def pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # mm_decomp will go through pad_mm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
        return torch.ops.aten.mm(mat1, mat2)[:, :-n_padded_length]
    else:
        mat1 = pad_dim(mat1, m_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)[:-m_padded_length, :]


@register_decomposition([aten.bmm])
def bmm_decomp(mat1, mat2):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.bmm)
    ):
        m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))

        if k_padded_length != 0 or n_padded_length != 0 or m_padded_length != 0:
            pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)

    return NotImplemented  # go directly to lowering


def pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # bmm_decomp will go through pad_bmm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 2)
        mat2 = pad_dim(mat2, k_padded_length, 1)
        return torch.ops.aten.bmm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 2)
        return torch.ops.aten.bmm(mat1, mat2)[:, :, :-n_padded_length].contiguous()
    else:
        mat1 = pad_dim(mat1, m_padded_length, 1)
        return torch.ops.aten.bmm(mat1, mat2)[:, :-m_padded_length, :].contiguous()


@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    if not output_mask[2] or grad_output.device.type != "cuda":
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)


@register_decomposition([aten.log2])
def log2(x):
    return torch.log(x) * (1.0 / math.log(2.0))


@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.rsub.Tensor, aten.rsub.Scalar])
def rsub(a, b):
    if isinstance(b, numbers.Number):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    return b - a


@register_decomposition([aten.nan_to_num])
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    if is_boolean_dtype(x.dtype) or is_integer_dtype(x.dtype):
        return x

    if nan is None:
        nan = 0.0
    if posinf is None:
        posinf = torch.finfo(x.dtype).max
    if neginf is None:
        neginf = torch.finfo(x.dtype).min
    nan, posinf, neginf = (
        torch.tensor(v, dtype=x.dtype, device=x.device) for v in (nan, posinf, neginf)
    )
    x = torch.where(x != x, nan, x)
    x = torch.where(x == float("inf"), posinf, x)
    x = torch.where(x == float("-inf"), neginf, x)
    return x


@register_decomposition([aten.all.default])
def all(input):
    return torch.logical_not(torch.any(torch.logical_not(input)))


@register_decomposition([aten.all.dim])
def all_dim(input, dim, keeepdim=False):
    return torch.logical_not(torch.any(torch.logical_not(input), dim, keeepdim))


# NB: this decomposition is not stride accurate, do not put it in the main
# library
@register_decomposition(aten.copy)
def copy(self, src, non_blocking=False):
    intermediate = src.to(self, non_blocking)
    if self.size() != intermediate.size():
        return aten.expand_copy.default(intermediate, self.size())
    else:
        return intermediate


@register_decomposition([aten.baddbmm])
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    return self + result


@register_decomposition([aten.conj_physical])
def conj_physical(self):
    assert not self.is_complex(), "TODO: implement this"
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self):
    return self


@register_decomposition([aten.bernoulli.default])
def bernoulli(self, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < self


@register_decomposition([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < p


"""
Some decomps result in differences from eager related to randomness.
We put these decomps in a separate table `extra_random_decomps` to allow
turning them on and off via `config.fallback_random`.
"""
extra_random_decomps = get_decompositions(
    [aten.native_dropout, aten.exponential, aten.exponential_, aten.uniform_]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)


@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()
