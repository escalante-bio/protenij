import protenix.backend as backend
import protenix
from jaxtyping import Bool, Float, Array, Int
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import math


import einops
from .backend import (
    from_torch,
    AbstractFromTorch,
    register_from_torch,
    Linear,
    LayerNorm,
)

import protenix
import protenix.model
import protenix.model.modules
import protenix.model.modules.primitives
import protenix.model.triangular
import protenix.model.triangular.layers
import protenix.model.triangular.triangular
import protenix.model.modules.transformer
import protenix.model.modules.pairformer
import protenix.model.modules.embedders
import protenix.model.modules.head
import protenix.model.modules.diffusion
import protenix.model.generator
import protenix.model.modules.confidence
import protenix.model.protenix

def move_final_dim_to_dim(x, dim: int):
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return jnp.transpose(x, new_order)  # x.permute(new_order)


def unfold(x, dim, size, step):
    dim_size = x.shape[dim]
    n_patches = (dim_size - size) // step + 1
    indices = jnp.arange(n_patches)[:, None] * step + jnp.arange(size)[None, :]
    x_moved = jnp.moveaxis(x, dim, -1)
    patches = x_moved[..., indices]
    patches = jnp.moveaxis(patches, -2, dim)
    if dim == -1:
        patches = jnp.swapaxes(patches, -2, -1)
    return patches


def pad_at_dim(
    x,
    dim: int,
    pad_length: tuple[int] | list[int],
    value: float = 0,
):
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = dim % n_dim
    pad_widths = [(0, 0)] * dim + [tuple(pad_length)] + [(0, 0)] * (n_dim - dim - 1)
    return jnp.pad(x, pad_widths, constant_values=value)


def reshape_at_dim(x, dim: int, target_shape):
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def optimized_concat_split(attn_bias, n_queries: int):
    """Optimized concatenation and splitting of attention bias tensor.

    Args:
        attn_bias (torch.Tensor): The attention bias tensor.
            Shape: [..., D, E]
        n_queries (int): The number of queries in each split.

    Returns:
        torch.Tensor: The reshaped and permuted attention bias tensor.
            Shape: [..., n_queries, D // n_queries * E]
    """
    D = attn_bias.shape[-2]
    E = attn_bias.shape[-1]
    assert D % n_queries == 0
    num_splits = D // n_queries
    reshaped = attn_bias.reshape(*attn_bias.shape[:-2], num_splits, n_queries, E)
    permuted = jnp.transpose(reshaped, (*range(reshaped.ndim - 3), -2, -3, -1))
    output = permuted.reshape(*attn_bias.shape[:-2], n_queries, num_splits * E)
    return output


def rearrange_qk_to_dense_trunk(
    q,
    k,
    dim_q,
    dim_k,
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
):
    assert n_keys >= n_queries
    assert n_queries & 0x01 == 0
    assert n_keys & 0x01 == 0

    def basic_checks(x, dim_x):
        if isinstance(x, list):
            x_is_list = True
            assert isinstance(dim_x, list)
        else:
            x_is_list = False
            x = [x]
            dim_x = [dim_x]
        n_x = x[0].shape[dim_x[0]]  # x[0].size(dim_x[0])
        for i in range(len(dim_x)):
            if dim_x[i] < 0:
                dim_x[i] = len(x[i].shape) + dim_x[i]
            assert x[i].shape[dim_x[i]] == n_x  # x[i].size(dim_x[i]) == n_x
        return x, dim_x, x_is_list, n_x, len(x)

    q, dim_q, q_is_list, n, num_q = basic_checks(q, dim_q)
    k, dim_k, k_is_list, n_k, num_k = basic_checks(k, dim_k)

    assert n == n_k
    n_trunks = int(math.ceil(n / n_queries))
    q_pad_length = n_trunks * n_queries - n

    q_new = [
        pad_at_dim(q[i], dim=dim_q[i], pad_length=(0, q_pad_length))
        for i in range(num_q)
    ]
    q_trunked = [
        reshape_at_dim(q_new[i], dim=dim_q[i], target_shape=(n_trunks, n_queries))
        for i in range(num_q)
    ]

    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1 / 2) * n_queries + n_keys / 2 - n + 1 / 2)

    k_new = [
        pad_at_dim(k[i], dim=dim_k[i], pad_length=(pad_left, pad_right))
        for i in range(num_k)
    ]
    k_trunked = [
        unfold(k_new[i], dim_k[i], size=n_keys, step=n_queries) for i in range(num_k)
    ]

    k_trunked = [
        move_final_dim_to_dim(k_trunked[i], dim=dim_k[i] + 1) for i in range(num_k)
    ]

    if compute_mask:
        pad_mask = jnp.ones(
            (
                *(1,) * len(q[0].shape[:-2]),
                n + q_pad_length,
                n + pad_left + pad_right,
            )
        )
        pad_mask = (
            pad_mask.at[..., :n, 0:pad_left]
            .set(0)
            .at[..., :n, pad_left + n : :]
            .set(0)
            .at[..., n::, :]
            .set(0)
        )

        concat_split_data = optimized_concat_split(pad_mask, n_queries)
        pad_mask_trunked = (
            jnp.swapaxes(
                unfold(concat_split_data, -1, n_keys, pad_mask.shape[-1] + n_queries),
                -2,
                -3,
            )
        ).astype(bool)
    else:
        pad_mask_trunked = None

    if not q_is_list:
        q_trunked = q_trunked[0]
    if not k_is_list:
        k_trunked = k_trunked[0]

    padding_info = {
        "mask_trunked": pad_mask_trunked,
        "q_pad": q_pad_length,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }

    return q_trunked, k_trunked, padding_info


register_from_torch(protenix.model.triangular.layers.OpenFoldLayerNorm)(
    backend.LayerNorm
)


@register_from_torch(protenix.model.modules.primitives.Transition)
class Transition(AbstractFromTorch):
    layernorm1: backend.LayerNorm
    linear_no_bias_a: backend.Linear
    linear_no_bias_b: backend.Linear
    linear_no_bias: backend.Linear

    def __call__(self, x: Float[Array, "... c"]) -> Float[Array, "... c"]:
        x = self.layernorm1(x)
        a = self.linear_no_bias_a(x)
        b = self.linear_no_bias_b(x)
        return self.linear_no_bias(jax.nn.silu(a) * b)


@register_from_torch(protenix.model.triangular.layers.Dropout)
class Dropout(eqx.Module):
    rate: float
    batch_dim: list[int]

    def __call__(
        self,
        x: Float[Array, "..."],
        *,
        key: jax.random.PRNGKey,
        deterministic: bool,
    ):
        if deterministic:
            return x

        shape = x.shape
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1

        bools = jax.random.bernoulli(key=key, p=self.rate, shape=shape)

        return x * bools * (1 / (1 - self.rate))

    @staticmethod
    def from_torch(d: protenix.model.triangular.layers.Dropout):
        return Dropout(d.r, d.batch_dim)


@register_from_torch(
    protenix.model.triangular.triangular.TriangleMultiplicativeUpdate
)
class TriangleMultiplication(AbstractFromTorch):
    linear_a_p: backend.Linear
    linear_a_g: backend.Linear
    linear_b_p: backend.Linear
    linear_b_g: backend.Linear
    layer_norm_in: backend.LayerNorm
    layer_norm_out: backend.LayerNorm
    linear_g: backend.Linear
    linear_z: backend.Linear
    sigmoid: any
    _outgoing: bool

    def __call__(
        self,
        z_in: Float[Array, "... N N C_z"],
        mask: Bool[Array, "... N N"] | None,
    ) -> Float[Array, "... N N C_z"]:
        #with jax.default_matmul_precision("float32"):
        if mask is None:
            mask = jnp.ones_like(z_in, shape=z_in.shape[:-1])

        mask = mask[..., None]

        z = self.layer_norm_in(z_in)
        a = self.linear_a_p(z) * mask * self.sigmoid(self.linear_a_g(z))
        b = self.linear_b_p(z) * mask * self.sigmoid(self.linear_b_g(z))

        ### permute_final_dims (2, 0, 1)
        if self._outgoing:
            a = einops.rearrange(a, "... a b c -> ... c a b")
            b = einops.rearrange(b, "... a b c -> ... c b a")
        else:
            a = einops.rearrange(a, "... a b c -> ... c b a")
            b = einops.rearrange(b, "... a b c -> ... c a b")
        p = a @ b
        # return p
        x = einops.rearrange(p, "... a b c -> ... b c a")

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        # Gate uses layer_norm'd z, not z_in (matches PyTorch)
        g = self.sigmoid(self.linear_g(z))

        return x * g


@register_from_torch(protenix.model.triangular.layers.Attention)
class Attention(AbstractFromTorch):
    c_q: int  # input dimension of query
    c_k: int  # input dimension of key
    c_v: int  # input dimension of value
    c_hidden: int  # per-head hidden dimension
    no_heads: int  # number of heads
    gating: bool  # whether to use gating
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: callable

    # TODO: Add mask? Instead of infs....
    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        biases: None | list[Float[Array, "... H Q K"]],
    ) -> Float[Array, "... Q C_v"]:
        # apply linear
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        # and reshape rearrange to heads (..., H, Q/K/V, C_hidden)
        q = einops.rearrange(
            q, "... Q (H C_hidden) -> ... H Q C_hidden", H=self.no_heads
        )
        k = einops.rearrange(
            k, "... K (H C_hidden) -> ... H K C_hidden", H=self.no_heads
        )
        v = einops.rearrange(
            v, "... V (H C_hidden) -> ... H V C_hidden", H=self.no_heads
        )

        # scale q
        q = q / np.sqrt(self.c_hidden)

        # compute attention
        a = jnp.einsum("... h q d, ... h k d -> ... h q k", q, k)
        # add pairwise biases
        # todo: not this.
        for bias in biases:
            a += bias

        a = jax.nn.softmax(a, axis=-1)

        a = jnp.einsum("... h q k, ... h k d -> ... h q d", a, v)
        # equivalent of o = o.transpose(-2, -3)
        o = einops.rearrange(a, "... H Q C_hidden -> ... Q H C_hidden")
        if self.linear_g is not None:
            g = jax.nn.sigmoid(self.linear_g(q_x))
            g = einops.rearrange(
                g, "... (H C_hidden) -> ... H C_hidden", H=self.no_heads
            )
            o = o * g

        o = einops.rearrange(o, "... Q H C -> ... Q (H C)")

        return self.linear_o(o)


@register_from_torch(
    protenix.model.triangular.triangular.TriangleAttention
)
class TriangleAttention(AbstractFromTorch):
    starting: bool
    layer_norm: backend.LayerNorm
    linear: Linear
    mha: Attention
    inf: float

    def __call__(
        self,
        x: Float[Array, "... I J C_in"],
        mask: Float[Array, "... I J"] | None,
    ) -> Float[Array, "... I J C_in"]:
        if mask is None:
            mask = jnp.ones_like(x, shape=x.shape[:-1])

        if not self.starting:
            x = einops.rearrange(x, "... I J C -> ... J I C")
            mask = einops.rearrange(mask, "... I J -> ... J I")

        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = einops.rearrange(self.linear(x), "... A B C -> ... C A B")
        # [*, 1, H, I, J]
        triangle_bias = einops.rearrange(triangle_bias, "... H I J -> ... 1 H I J")

        x = self.mha(q_x=x, kv_x=x, biases=[mask_bias, triangle_bias])

        if not self.starting:
            x = einops.rearrange(x, "... I J C -> ... J I C")

        return x


def _attention(
    q,
    k,
    v,
    attn_bias=None,
):
    # NOTE: Do we use scaled attention here or not?
    assert k.shape == v.shape
    out = jax.nn.dot_product_attention(
        query=jnp.swapaxes(q, -3, -2),
        key=jnp.swapaxes(k, -3, -2),
        value=jnp.swapaxes(v, -3, -2),
        # bias=jnp.swapaxes(attn_bias, -1, 1),
        bias=attn_bias,
        scale=1.0,
    )
    return jnp.swapaxes(out, -3, -2)


@register_from_torch(protenix.model.modules.primitives.Attention)
class ProtenixAttention(AbstractFromTorch):
    c_q: int  # input dimension of query
    c_k: int  # input dimension of key
    c_v: int  # input dimension of value
    c_hidden: int  # per-head hidden dimension
    gating: bool  # whether to use gating
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: callable
    num_heads: int
    local_attention_method: str

    def _prep_qkv(self, q_x, kv_x):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = einops.rearrange(
            q, "... Q (H C_hidden) -> ... H Q C_hidden", H=self.num_heads
        )
        k = einops.rearrange(
            k, "... K (H C_hidden) -> ... H K C_hidden", H=self.num_heads
        )
        v = einops.rearrange(
            v, "... V (H C_hidden) -> ... H V C_hidden", H=self.num_heads
        )

        q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o, q_x):
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, G/Q, H, C_hidden]
            # g = g.view(g.shape[:-1] + (self.num_heads, -1))
            g = einops.rearrange(g, "... (H D) -> ... H D", H=self.num_heads)
            o = o * g

        # [*, Q, H * C_hidden]
        o = einops.rearrange(o, "... H D -> ... (H D)")

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    

    @staticmethod
    def rearrange_to_dense_trunk(
        q,
        k,
        v,
        n_queries: int,
        n_keys: int,
        attn_bias=None,
        inf: float = 1e10,
    ):
        assert n_keys >= n_queries
        assert n_queries & 0x01 == 0
        assert n_keys & 0x01 == 0

        n, d = q.shape[-2:]

        q_trunked, kv_trunked, padding_info = (
            rearrange_qk_to_dense_trunk(
                q=q,
                k=[k, v],
                dim_q=-2,
                dim_k=[-2, -2],
                n_queries=n_queries,
                n_keys=n_keys,
                compute_mask=False,
            )
        )
        q_pad_length, pad_left, pad_right = (
            padding_info["q_pad"],
            padding_info["k_pad_left"],
            padding_info["k_pad_right"],
        )

        # Padded_width = n + pad_left + pad_right
        if attn_bias is None:

            attn_bias = jnp.zeros(
                (
                    *(1,) * len(q.shape[:-2]),
                    n + q_pad_length,
                    n + pad_left + pad_right,
                )
            )
            attn_bias = attn_bias.at[..., :n, 0:pad_left].set(-inf)
            attn_bias = attn_bias.at[..., :n, pad_left + n : :].set(-inf)
            attn_bias = attn_bias.at[..., n::, :].set(-inf)

        else:
            assert False
            # attn_bias = F.pad(
            #     attn_bias, (pad_left, pad_right, 0, q_pad_length), value=-inf
            # )

        concat_split_data = optimized_concat_split(attn_bias, n_queries)
        # attn_bias_trunked = unfold(
        #     concat_split_data, -1, n_keys, attn_bias.shape[-1] + n_queries
        # ).transpose(-2, -3)
        attn_bias_trunked = jnp.swapaxes(
            unfold(concat_split_data, -1, n_keys, attn_bias.shape[-1] + n_queries),
            -2,
            -3,
        )
        return (
            q_trunked,
            kv_trunked[0],
            kv_trunked[1],
            attn_bias_trunked,
            q_pad_length,
        )

    @staticmethod
    def _local_attention(
        *,
        q,
        k,
        v,
        n_queries: int,
        n_keys: int,
        attn_bias=None,
        trunked_attn_bias=None,
        inf: float = 1e10,
    ):
        assert (
            q.shape == k.shape == v.shape
        )  # local attention doesn't make sense if Q != K

        # Prepare for attention qkv, q: [..., n_trunks, n_queries, d], kv: [..., n_trunks, n_keys, d]

        # Rerrange to dense trunks
        # q: [*, n, d] -> [*, n_trunks, n_queries, d]
        # kv: [*, n, d] -> [*, n_trunks, n_keys, d]
        # attn_bias: [*, n, d] -> [*, n_trunks, n_queries, n_keys]
        q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_pad_length = (
            ProtenixAttention.rearrange_to_dense_trunk(
                q=q,
                k=k,
                v=v,
                n_queries=n_queries,
                n_keys=n_keys,
                attn_bias=attn_bias,
                inf=inf,
            )
        )

        # Apply attention
        # [..., n_trunks, n_queries, d]
        if trunked_attn_bias is not None:
            attn_bias_trunked = attn_bias_trunked + trunked_attn_bias

        # if we have an extra batch dimension do a vmap...
        q_size = q_trunked.shape
        if len(q_size) == 5:
            out = jax.vmap(lambda q,k,v,ab: jax.nn.dot_product_attention(
                query=jnp.swapaxes(q, -3, -2),
                key=jnp.swapaxes(k, -3, -2),
                value=jnp.swapaxes(v, -3, -2),
                bias=ab,  # jnp.swapaxes(attn_bias_trunked, -1, 1),
                scale=1.0,
            ))(q_trunked, k_trunked, v_trunked, attn_bias_trunked)
        else:
            out = jax.nn.dot_product_attention(
                query=jnp.swapaxes(q_trunked, -3, -2),
                key=jnp.swapaxes(k_trunked, -3, -2),
                value=jnp.swapaxes(v_trunked, -3, -2),
                bias=attn_bias_trunked,  # jnp.swapaxes(attn_bias_trunked, -1, 1),
                # XXX TODO: do we need a swapaxes here?
                scale=1.0,
            )
        out = jnp.swapaxes(out, -3, -2)

        # Revert back to orignal shape and remove q_pad_length
        # [..., n_trunks, n_queries, d] ->  [..., n_trunks * n_queries, d] ->  [..., n, d]
        out = out.reshape(*out.shape[:-3], -1, out.shape[-1])
        if q_pad_length > 0:
            out = out[..., :-q_pad_length, :]
        return out

    # TODO: Add mask? Instead of infs....
    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        attn_bias=None,
        trunked_attn_bias=None,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inf: float | None = 1e10,
    ) -> Float[Array, "... Q C_v"]:
        assert self.local_attention_method == "local_cross_attention"

        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x)

        if attn_bias is not None:
            if len(attn_bias.shape) == len(q.shape):
                assert attn_bias.shape[:-2] == q.shape[:-2]
            else:
                assert len(attn_bias.shape) == len(q.shape) - 1
                assert attn_bias.shape[:-2] == q.shape[:-3]
                # Expand at head dim, got shape [..., 1, Q, K]
                attn_bias = attn_bias[..., None, :, :]

        if trunked_attn_bias is not None:
            # NOTE: trunked_attn_bias can only be used with "local_cross_attention" method
            assert n_queries and n_keys
            assert self.local_attention_method == "local_cross_attention"

            if len(trunked_attn_bias.shape) == len(q.shape) + 1:
                assert trunked_attn_bias.shape[:-3] == q.shape[:-2]
            else:
                assert len(trunked_attn_bias.shape) == len(q.shape)
                # Expand at head dim, got shape [..., 1, n_trunks, n_queries, n_keys]
                trunked_attn_bias = trunked_attn_bias[
                    ..., None, :, :, :
                ]  # trunked_attn_bias.unsqueeze(dim=-4)

        if n_queries and n_keys:
            o = self._local_attention(
                q=q,
                k=k,
                v=v,
                n_queries=n_queries,
                n_keys=n_keys,
                attn_bias=attn_bias,
                trunked_attn_bias=trunked_attn_bias,
                inf=inf,
            )
        else:
            o = _attention(
                q=q,
                k=k,
                v=v,
                attn_bias=attn_bias,
            )  # [*, H, Q, C_hidden]

        o = einops.rearrange(o, "... H Q C -> ... Q H C")

        return self._wrap_up(o, q_x)


@register_from_torch(protenix.model.modules.primitives.AdaptiveLayerNorm)
class AdaptiveLayerNorm(AbstractFromTorch):
    layernorm_a: backend.LayerNorm
    layernorm_s: backend.LayerNorm
    linear_s: Linear
    linear_nobias_s: Linear

    def __call__(self, a, s):
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = jax.nn.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


@register_from_torch(protenix.model.modules.transformer.AttentionPairBias)
class AttentionPairBias(AbstractFromTorch):
    layernorm_a: AdaptiveLayerNorm | LayerNorm
    layernorm_kv: AdaptiveLayerNorm | LayerNorm | None
    layernorm_z: LayerNorm
    attention: ProtenixAttention
    linear_nobias_z: Linear
    has_s: bool
    cross_attention_mode: bool
    linear_a_last: Linear | None

    def local_multihead_attention(
        self,
        q,
        kv,
        z,
        n_queries: int = 32,
        n_keys: int = 128,
    ):
        assert n_queries == z.shape[-3]
        assert n_keys == z.shape[-2]
        assert len(z.shape) == len(q.shape) + 2

        # Multi-head attention bias
        bias = self.linear_nobias_z(
            self.layernorm_z(z)
        )  # [..., n_blocks, n_queries, n_keys, n_heads]
        # bias = permute_final_dims(
        #     bias, [3, 0, 1, 2]
        # )  # [..., n_heads, n_blocks, n_queries, n_keys]
        bias = einops.rearrange(
            bias, "... A B C D -> ... D A B C"
        )  # [..., n_heads, n_blocks, n_queries, n_keys]

        # Line 11: Multi-head attention with attention bias & gating (and optionally local attention)

        q = self.attention(
            q_x=q,
            kv_x=kv,
            trunked_attn_bias=bias,
            n_queries=n_queries,
            n_keys=n_keys,
        )
        return q

    def __call__(
        self,
        a,
        s,
        z,
        n_queries: int | None = None,
        n_keys: int | None = None,
    ):
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a=a, s=s)
            else:
                kv = self.layernorm_kv(a)
        else:
            kv = None

        # Multihead attention with pair bias
        if n_queries and n_keys:
            a = self.local_multihead_attention(
                a,
                kv if self.cross_attention_mode else a,
                z,
                n_queries,
                n_keys,
            )
        else:
            bias = self.linear_nobias_z(self.layernorm_z(z))
            bias = einops.rearrange(bias, "... A B C -> ... C A B")
            a = self.attention(
                a, kv if self.cross_attention_mode else a, attn_bias=bias
            )

        # Output projection (from adaLN-Zero [27])
        if self.has_s:
            a *= jax.nn.sigmoid(self.linear_a_last(s))

        return a


@register_from_torch(protenix.model.modules.pairformer.PairformerBlock)
class PairformerBlock(AbstractFromTorch):
    tri_mul_out: TriangleMultiplication
    tri_mul_in: TriangleMultiplication
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    dropout_row: Dropout
    pair_transition: Transition
    attention_pair_bias: AttentionPairBias | None
    single_transition: Transition | None
    c_s: int

    def __call__(self, *, s, z, pair_mask, key):
        # TODO: Dropout?!
        z += self.tri_mul_out(
            z,
            mask=pair_mask,
        )
        z += self.tri_mul_in(
            z,
            mask=pair_mask,
        )
        z += self.tri_att_start(
            z,
            mask=pair_mask,
        )
        z = jnp.swapaxes(z, -2, -3)
        z += self.tri_att_end(
            z,
            mask=jnp.swapaxes(pair_mask, -1, -2) if pair_mask is not None else None,
        )
        z = jnp.swapaxes(z, -2, -3)
        z += self.pair_transition(z)

        if self.c_s > 0:
            s = s + self.attention_pair_bias(
                a=s,
                s=None,
                z=z,
            )
            s = s + self.single_transition(s)
        return s, z


@register_from_torch(protenix.model.modules.pairformer.PairformerStack)
class Pairformer(eqx.Module):
    stacked_parameters: PairformerBlock
    static: PairformerBlock

    @staticmethod
    def from_torch(m: protenix.model.modules.pairformer.PairformerStack):
        layers = [from_torch(layer) for layer in m.blocks]
        if len(layers) == 0:
            return Pairformer(None, None)
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return Pairformer(
            jax.tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )

    def __call__(self, s, z, pair_mask, key):
        @jax.checkpoint
        def body_fn(embedding, params):
            s, z, key = embedding
            s, z = eqx.combine(self.static, params)(
                s=s, z=z, pair_mask=pair_mask, key=key
            )
            return (s, z, key), None

        return jax.lax.scan(body_fn, (s, z, key), self.stacked_parameters)[0][:2]


@register_from_torch(protenix.model.triangular.layers.OuterProductMean)
class OuterProductMean(AbstractFromTorch):
    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: Float

    def __call__(
        self,
        m: Float[Array, "... N_seq N_res C_m"],
        # mask: Bool[Array, "... N_seq N_res"],
    ):
        # if mask is None:
        mask = jnp.ones(m.shape[:-1])  # , dtype=bool)
        mask = mask.astype(jnp.float32)
        ln = self.layer_norm(m)

        mask = mask[..., None]
        a = mask * self.linear_1(ln)

        b = mask * self.linear_2(ln)

        a = jnp.swapaxes(a, -2, -3)
        b = jnp.swapaxes(b, -2, -3)

        outer = jnp.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)

        # [*, N_res, N_res, 1]
        norm = jnp.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        return outer / norm


@register_from_torch(protenix.model.modules.pairformer.MSAPairWeightedAveraging)
class MSAPairWeightedAveraging(AbstractFromTorch):
    c_m: int
    c: int
    n_heads: int
    c_z: int
    layernorm_m: LayerNorm
    linear_no_bias_mv: Linear
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    linear_no_bias_mg: Linear
    softmax_w: any
    linear_no_bias_out: Linear

    def __call__(self, m, z):
        m = self.layernorm_m(m)
        v = self.linear_no_bias_mv(m)
        v = v.reshape(*v.shape[:-1], self.n_heads, self.c)
        b = self.linear_no_bias_z(self.layernorm_z(z))
        w = self.softmax_w(b)
        o = jnp.einsum("...ijh,...sjhc->...sihc", w, v)
        o = o.reshape(*o.shape[:-2], self.n_heads * self.c)
        g = jax.nn.sigmoid(self.linear_no_bias_mg(m))
        m = self.linear_no_bias_out(g * o)
        return m


@register_from_torch(protenix.model.modules.pairformer.MSAStack)
class MSAStack(AbstractFromTorch):
    c: int
    msa_pair_weighted_averaging: MSAPairWeightedAveraging
    dropout_row: any
    transition_m: Transition

    def __call__(self, m, z, *, key):
        m = m + self.msa_pair_weighted_averaging(m, z)
        m = m + self.transition_m(m)
        return m


@register_from_torch(protenix.model.modules.pairformer.MSABlock)
class MSABlock(eqx.Module):
    c_m: int
    c_z: int
    outer_product_mean_msa: OuterProductMean
    msa_stack: MSAStack | None
    pair_stack: PairformerBlock

    @staticmethod
    def from_torch(m: protenix.model.modules.pairformer.MSABlock):
        msa_stack = from_torch(m.msa_stack) if hasattr(m, "msa_stack") else None
        return MSABlock(
            c_m=m.c_m,
            c_z=m.c_z,
            outer_product_mean_msa=from_torch(m.outer_product_mean_msa),
            msa_stack=msa_stack,
            pair_stack=from_torch(m.pair_stack),
        )

    def __call__(self, m, z, pair_mask, *, key):
        z = z + self.outer_product_mean_msa(m)
        if self.msa_stack is not None:
            m = self.msa_stack(m, z, key=key)
        _, z = self.pair_stack(s=None, z=z, pair_mask=pair_mask, key=key)
        return m, z


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, any],
    *,
    cutoff: int = 512,
    key,
) -> dict[str, any]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].shape[-2]
    cutoff = min(cutoff, msa_len)
    indices = jax.random.choice(key=key, a=msa_len, replace=False, shape=(cutoff,))
    # XXX TODO: by default Protenix will pick a random number of sequences, we ignore this!
    return jax.tree.map(lambda v: jnp.take(v, indices=indices, axis=-2), feat_dict)


@register_from_torch(protenix.model.modules.pairformer.MSAModule)
class MSAModule(eqx.Module):
    linear_no_bias_m: Linear
    linear_no_bias_s: Linear
    # MSABlocks have heterogeneous tree structure (some have msa_stack, some don't)
    # so we store them as a list rather than trying to stack for scan.
    # With only 1-4 blocks, the JIT overhead is minimal.
    blocks: list[MSABlock]
    msa_configs: dict
    training: bool
    input_feature: dict
    # See https://github.com/jax-ml/jax/issues/24398
    input_feature_keys_ordered: list[str]

    def __call__(self, input_feature_dict: dict, z, s_inputs, pair_mask, *, key):
        if "msa" not in input_feature_dict:
            print("no msa in features")
            return z

        msa_feat = sample_msa_feature_dict_random_without_replacement(
            key=key,
            feat_dict={
                k: input_feature_dict[k]
                for k in ["msa", "has_deletion", "deletion_value"]
            },
            cutoff=(
                self.msa_configs["train_cutoff"]
                if self.training
                else self.msa_configs["test_cutoff"]
            ),
        )

        msa_feat["msa"] = jax.nn.one_hot(msa_feat["msa"], 32)
        target_shape = msa_feat["msa"].shape[:-1]

        msa_sample = jnp.concatenate(
            [
                msa_feat[k].reshape(*target_shape, self.input_feature[k])
                for k in self.input_feature_keys_ordered
            ],
            axis=-1,
        )
        msa_sample = self.linear_no_bias_m(msa_sample)
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)

        for block in self.blocks:
            key = jax.random.fold_in(key, 1)
            msa_sample, z = block(msa_sample, z, pair_mask, key=key)

        return z

    @staticmethod
    def from_torch(
        m: protenix.model.modules.pairformer.MSAModule,
    ):
        blocks = [from_torch(block) for block in m.blocks]
        return MSAModule(
            linear_no_bias_m=from_torch(m.linear_no_bias_m),
            linear_no_bias_s=from_torch(m.linear_no_bias_s),
            blocks=blocks,
            msa_configs=m.msa_configs,
            training=m.training,
            input_feature=from_torch(m.input_feature),
            input_feature_keys_ordered=list(m.input_feature.keys()),
        )


@register_from_torch(protenix.model.modules.transformer.ConditionedTransitionBlock)
class ConditionedTransitionBlock(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    linear_nobias_a1: Linear
    linear_nobias_a2: Linear
    linear_nobias_b: Linear
    linear_s: Linear

    def __call__(
        self, a: Float[Array, "... N C_a"], s: Float[Array, "... N C_s"]
    ) -> Float[Array, "... N C_a"]:
        a = self.adaln(a, s)
        a = self.linear_nobias_a2(a) * jax.nn.silu(self.linear_nobias_a1(a))
        a = jax.nn.sigmoid(self.linear_s(s)) * self.linear_nobias_b(a)
        return a


@register_from_torch(protenix.model.modules.transformer.DiffusionTransformerBlock)
class DiffusionTransformerBlock(AbstractFromTorch):
    n_heads: int
    c_a: int
    c_s: int
    c_z: int
    attention_pair_bias: AttentionPairBias
    conditioned_transition_block: ConditionedTransitionBlock
    drop_path: backend.Identity

    def __call__(
        self,
        a: Float[Array, "... N C_a"],
        s: Float[Array, "... N C_s"],
        z: Float[Array, "... N N C_z"],
        n_queries: int | None = None,
        n_keys: int | None = None,
    ) -> tuple[Float[Array, "... N C_a"], Float[Array, "... N N C_z"]]:
        # Apply attention pair bias
        attn_out = a + self.drop_path(
            self.attention_pair_bias(
                a=a,
                s=s,
                z=z,
                n_queries=n_queries,
                n_keys=n_keys,
            )
        )

        # Apply conditioned transition block
        ff_out = self.drop_path(self.conditioned_transition_block(a=attn_out, s=s))

        out_a = attn_out + ff_out

        return out_a, s, z


@register_from_torch(protenix.model.modules.transformer.DiffusionTransformer)
class DiffusionTransformer(eqx.Module):
    block_params: DiffusionTransformerBlock
    block_static: DiffusionTransformerBlock

    @staticmethod
    def from_torch(m: protenix.model.modules.transformer.DiffusionTransformer):
        layers = [from_torch(layer) for layer in m.blocks]
        _, static = eqx.partition(layers[0], eqx.is_inexact_array)
        return DiffusionTransformer(
            jax.tree.map(
                lambda *v: jnp.stack(v, 0),
                *[eqx.filter(layer, eqx.is_inexact_array) for layer in layers],
            ),
            static,
        )

    def __call__(
        self,
        a: Float[Array, "... N C_a"],
        s: Float[Array, "... N C_s"],
        z: Float[Array, "... N N C_z"],
        n_queries: int | None = None,
        n_keys: int | None = None,
    ):
        @jax.checkpoint
        def body_fn(embedding, params):
            a, s, z = embedding
            a, s, z = eqx.combine(self.block_static, params)(
                a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys
            )
            return (a, s, z), None

        return jax.lax.scan(body_fn, (a, s, z), self.block_params)[0][0]


@register_from_torch(protenix.model.modules.transformer.AtomTransformer)
class AtomTransformer(AbstractFromTorch):
    diffusion_transformer: DiffusionTransformer
    n_queries: int
    n_keys: int

    def __call__(self, q, c, p):
        n_blocks, n_queries, n_keys = p.shape[-4:-1]
        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(
            a=q,
            s=c,
            z=p,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
        )


def average_over_atoms(
    x_atom: Float[Array, "... N_atom D"],
    atom_to_token_idx: Int[Array, "... N_atom"],
    n_res: int,
):
    
    def _helper(x_atom):
        assert x_atom.ndim == 2
        assert atom_to_token_idx.ndim == 1
        n_atoms = x_atom.shape[0]
        d = x_atom.shape[1]

        def body_function(accumulated, T):
            residue_index, atom_value = T
            return accumulated.at[residue_index].add(atom_value), None

        atoms_per_residue_count = jax.lax.scan(
            body_function,
            init=jnp.zeros((n_res,)),
            xs=(atom_to_token_idx, jnp.ones(n_atoms)),
        )[0]

        accumulated_sum = jax.lax.scan(
            body_function, init=jnp.zeros((n_res, d)), xs=(atom_to_token_idx, x_atom)
        )[0]

        return accumulated_sum / atoms_per_residue_count[..., None]
    
    n_batch_dim = len(x_atom.shape) - 2
    f = _helper
    for _ in range(n_batch_dim):
        f = jax.vmap(f)
    return f(x_atom)


def gather_pair_embedding_in_dense_trunk(x, idx_q, idx_k):
    """
    Selectively gather elements from a tensor using two sets of indices.

        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q]
        idx_k: [N_b, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]
    """
    idx_q = idx_q
    idx_k = idx_k
    assert len(idx_q.shape) == len(idx_k.shape) == 2

    # Get the shape parameters
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]

    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = jnp.tile(idx_q[..., None], (1, 1, N_k))  # .expand(-1, -1, N_k)
    idx_k_expanded = jnp.tile(idx_k[:, None, ...], (1, N_q, 1))  # unsqueeze(1)?

    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]

    return y


def broadcast_token_to_local_atom_pair(
    z_token,
    atom_to_token_idx,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
):
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (torch.Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (torch.Tensor): map atom idx to token idx
            [N_atom]

    Returns:
        z_gathered_blocked (torch.Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
        pad_mask (torch.Tensor):
            [n_trunks, n_queries, n_keys]
        q_pad_length (int)
    """

    # [N_atom] -> [n_trunks, n_queries] and [n_trunks, n_keys]
    atom_to_token_idx_q, atom_to_token_idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )

    z_gathered_blocked = gather_pair_embedding_in_dense_trunk(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked, pad_info


@register_from_torch(protenix.model.modules.transformer.AtomAttentionEncoder)
class AtomAttentionEncoder(eqx.Module):
    has_coords: bool
    c_atom: int
    c_atompair: int
    c_token: int
    c_s: int
    c_z: int
    n_queries: int
    n_keys: int
    input_feature: dict[str, int]
    input_feature_keys_ordered: list[str]
    linear_no_bias_ref_pos: Linear
    linear_no_bias_ref_charge: Linear
    linear_no_bias_f: Linear
    linear_no_bias_d: Linear
    linear_no_bias_invd: Linear
    linear_no_bias_v: Linear

    layernorm_s: LayerNorm | None
    linear_no_bias_s: Linear | None
    layernorm_z: LayerNorm | None
    linear_no_bias_z: Linear | None
    linear_no_bias_r: Linear | None

    linear_no_bias_cl: Linear
    linear_no_bias_cm: Linear
    small_mlp: backend.Sequential
    atom_transformer: AtomTransformer
    linear_no_bias_q: Linear

    def __call__(self, input_feature_dict: dict[str], r_l=None, s=None, z=None):
        if self.has_coords:
            assert r_l is not None
            assert s is not None
            assert z is not None

        atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
        # Create the atom single conditioning: Embed per-atom meta data
        # [..., N_atom, C_atom]
        batch_shape = input_feature_dict["ref_pos"].shape[:-2]
        N_atom = input_feature_dict["ref_pos"].shape[-2]
        c_l = self.linear_no_bias_ref_pos(
            input_feature_dict["ref_pos"]
        ) + self.linear_no_bias_ref_charge(
            # use arcsinh for ref_charge
            jnp.arcsinh(input_feature_dict["ref_charge"]).reshape(
                *batch_shape, N_atom, 1
            )
        )

        # ORDERED DICTIONARY.. https://github.com/jax-ml/jax/issues/24398
        c_l = c_l + self.linear_no_bias_f(
            jnp.concatenate(
                [
                    input_feature_dict[name].reshape(
                        *batch_shape, N_atom, self.input_feature[name]
                    )
                    for name in self.input_feature_keys_ordered
                ],
                axis=-1,
            )
        )
        c_l = c_l * input_feature_dict["ref_mask"].reshape(*batch_shape, N_atom, 1)

        # Line2-Line4: Embed offsets between atom reference positions

        # Prepare tensors in dense trunks for local operations

        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=True,
        )

        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].astype(jnp.int32)
            == k_trunked_list[1][..., None, :].astype(jnp.int32)
        )[..., None]  # [..., n_blocks, n_queries, n_keys, 1]
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info["mask_trunked"][
            ..., None
        ]  # [..., n_blocks, n_queries, n_keys, C_atompair]

        # Line5-Line6: Embed pairwise inverse squared distances, and the valid mask

        p_lm = (
            p_lm
            + self.linear_no_bias_invd(1 / (1 + (d_lm**2).sum(axis=-1, keepdims=True)))
            * v_lm
        )
        p_lm = p_lm + self.linear_no_bias_v(v_lm)  # not multipling v_lm

        # Line7: Initialise the atom single representation as the single conditioning
        # q_l = c_l.clone()

        # If provided, add trunk embeddings and noisy positions
        n_token = None
        if r_l is not None:
            # assert False
            # N_sample = r_l.shape[-3]

            # Broadcast the single and pair embedding from the trunk
            # n_token = s.shape[-2]
            c_l = (
                c_l[..., None, :, :]
                + broadcast_token_to_atom(  # c_l.unsqueeze(dim=-3) + broadcast_token_to_atom(
                    x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                    atom_to_token_idx=atom_to_token_idx,
                )
            )  # [..., N_sample, N_atom, c_atom]

            p_lm = (
                p_lm[..., None, :, :, :, :]  # p_lm.unsqueeze(dim=-5)
                + broadcast_token_to_local_atom_pair(
                    z_token=self.linear_no_bias_z(self.layernorm_z(z)),
                    atom_to_token_idx=atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )[0]
            )  # [..., N_sample, n_blocks, n_queries, n_keys, c_atompair]

            # Add the noisy positions
            # Different from paper!!
            q_l = c_l + self.linear_no_bias_r(r_l)  # [..., N_sample, N_atom, c_atom]
        else:
            q_l = c_l  # .clone()

        # Add the combined single conditioning to the pair representation
        c_l_q, c_l_k, _ = rearrange_qk_to_dense_trunk(
            q=c_l,
            k=c_l,
            dim_q=-2,
            dim_k=-2,
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        # if inplace_safe:
        p_lm += self.linear_no_bias_cl(jax.nn.relu(c_l_q[..., None, :]))
        p_lm += self.linear_no_bias_cm(jax.nn.relu(c_l_k[..., None, :, :]))
        p_lm += self.small_mlp(p_lm)

        # Cross attention transformer

        q_l = self.atom_transformer(q_l, c_l, p_lm)  # [..., (N_sample), N_atom, c_atom]

        # Aggregate per-atom representation to per-token representation
        a = average_over_atoms(
            x_atom=jax.nn.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_res=input_feature_dict["residue_index"].shape[-1],
        )


        #     torch.cuda.empty_cache()
        return a, q_l, c_l, p_lm

    @staticmethod
    def from_torch(m: protenix.model.modules.transformer.AtomAttentionEncoder):
        return AtomAttentionEncoder(
            has_coords=m.has_coords,
            c_atom=m.c_atom,
            c_atompair=m.c_atompair,
            c_token=m.c_token,
            c_s=m.c_s,
            c_z=m.c_z,
            n_queries=m.n_queries,
            n_keys=m.n_keys,
            input_feature=from_torch(m.input_feature),
            input_feature_keys_ordered=list(m.input_feature.keys()),
            linear_no_bias_ref_pos=from_torch(m.linear_no_bias_ref_pos),
            linear_no_bias_ref_charge=from_torch(m.linear_no_bias_ref_charge),
            linear_no_bias_f=from_torch(m.linear_no_bias_f),
            linear_no_bias_d=from_torch(m.linear_no_bias_d),
            linear_no_bias_invd=from_torch(m.linear_no_bias_invd),
            linear_no_bias_v=from_torch(m.linear_no_bias_v),
            layernorm_s=from_torch(m.layernorm_s)
            if hasattr(m, "layernorm_s")
            else None,
            linear_no_bias_s=from_torch(m.linear_no_bias_s)
            if hasattr(m, "linear_no_bias_s")
            else None,
            layernorm_z=from_torch(m.layernorm_z)
            if hasattr(m, "layernorm_z")
            else None,
            linear_no_bias_z=from_torch(m.linear_no_bias_z)
            if hasattr(m, "linear_no_bias_z")
            else None,
            linear_no_bias_r=from_torch(m.linear_no_bias_r)
            if hasattr(m, "linear_no_bias_r")
            else None,
            linear_no_bias_cl=from_torch(m.linear_no_bias_cl),
            linear_no_bias_cm=from_torch(m.linear_no_bias_cm),
            small_mlp=from_torch(m.small_mlp),
            atom_transformer=from_torch(m.atom_transformer),
            linear_no_bias_q=from_torch(m.linear_no_bias_q),
        )


@register_from_torch(protenix.model.modules.embedders.InputFeatureEmbedder)
class InputFeatureEmbedder(eqx.Module):
    c_atom: int
    c_atompair: int
    c_token: int
    atom_attention_encoder: AtomAttentionEncoder
    input_feature: dict[str, int]
    input_feature_keys_ordered: list[str]

    def __call__(self, input_feature_dict: dict[str]):
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = jnp.concatenate(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, self.input_feature[name])
                for name in self.input_feature_keys_ordered
            ],
            axis=-1,
        )

        return s_inputs

    @staticmethod
    def from_torch(
        m: protenix.model.modules.embedders.InputFeatureEmbedder,
    ):
        return InputFeatureEmbedder(
            c_atom=m.c_atom,
            c_atompair=m.c_atompair,
            c_token=m.c_token,
            atom_attention_encoder=from_torch(m.atom_attention_encoder),
            input_feature=from_torch(m.input_feature),
            input_feature_keys_ordered=list(m.input_feature.keys()),
        )


@register_from_torch(protenix.model.modules.embedders.RelativePositionEncoding)
class RelativePositionEncoding(AbstractFromTorch):
    r_max: int
    s_max: int
    c_z: int
    linear_no_bias: Linear
    input_feature: dict[str, int]

    def __call__(self, input_feature_dict: dict[str, jnp.ndarray]):
        b_same_chain = (
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).astype(jnp.int32)  # [..., N_token, N_token]
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).astype(jnp.int32)  # [..., N_token, N_token]
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
        ).astype(jnp.int32)  # [..., N_token, N_token]
        d_residue = jnp.clip(
            input_feature_dict["residue_index"][..., :, None]
            - input_feature_dict["residue_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_pos = jax.nn.one_hot(d_residue, 2 * (self.r_max + 1))

        d_token = jnp.clip(
            input_feature_dict["token_index"][..., :, None]
            - input_feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = jax.nn.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = jnp.clip(
            input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = jax.nn.one_hot(d_chain, 2 * (self.s_max + 1))
        p = self.linear_no_bias(
            jnp.concatenate(
                [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                axis=-1,
            )
        )
        return p


@register_from_torch(protenix.model.modules.head.DistogramHead)
class DistogramHead(AbstractFromTorch):
    linear: Linear

    def __call__(self, z):
        logits = self.linear(z)
        logits = logits + jnp.swapaxes(logits, -2, -3)
        return logits


@register_from_torch(protenix.model.modules.embedders.FourierEmbedding)
class FourierEmbedding(AbstractFromTorch):
    w: Float[Array, "..."]
    b: Float[Array, "..."]

    def __call__(self, t_hat_noise_level):
        return jnp.cos(2 * jnp.pi * (self.w * t_hat_noise_level[..., None] + self.b))


@register_from_torch(protenix.model.modules.diffusion.DiffusionConditioning)
class DiffusionConditioning(AbstractFromTorch):
    sigma_data: float
    c_z: int
    c_s: int
    c_s_inputs: int
    relpe: RelativePositionEncoding
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    transition_z1: Transition
    transition_z2: Transition

    layernorm_s: LayerNorm
    linear_no_bias_s: Linear
    fourier_embedding: FourierEmbedding
    layernorm_n: LayerNorm
    linear_no_bias_n: Linear
    transition_s1: Transition
    transition_s2: Transition

    def __call__(
        self,
        t_hat_noise_level,
        input_feature_dict: dict,
        s_inputs,
        s_trunk,
        z_trunk,
        use_conditioning=False,
    ):
        if not use_conditioning:
            s_trunk = 0 * s_trunk
            z_trunk = 0 * z_trunk

        pair_z = jnp.concatenate(
            [z_trunk, self.relpe(input_feature_dict)], axis=-1
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))

        pair_z += self.transition_z1(pair_z)
        pair_z += self.transition_z2(pair_z)

        # Single conditioning
        single_s = jnp.concatenate(
            [s_trunk, s_inputs], axis=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=jnp.log(t_hat_noise_level / self.sigma_data) / 4
        )
        single_s = (
            single_s[..., None, :, :]
            + self.linear_no_bias_n(  # single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
                self.layernorm_n(noise_n)
            )[..., None, :]
        )  # .unsqueeze(
        #  dim=-2

        single_s += self.transition_s1(single_s)
        single_s += self.transition_s2(single_s)

        return single_s, pair_z


def broadcast_token_to_atom(
    x_token: Float[Array, "1 N_token C_token"],
    atom_to_token_idx: Int[Array, "1 N_atom"],
) -> Float[Array, "1 N_atom C_token"]:
    """Broadcast per-token activations to per-atom activations."""
    assert len(atom_to_token_idx.shape) == 1
    # assert x_token.ndim == 3, "x_token should have shape [..., N_token, C_token]"
    # assert x_token.shape[0] == 1, "x_token should have a batch dimension of size 1"
    out = x_token[..., atom_to_token_idx, :]

    return out


@register_from_torch(protenix.model.modules.transformer.AtomAttentionDecoder)
class AtomAttentionDecoder(AbstractFromTorch):
    n_blocks: int
    n_heads: int
    c_token: int
    c_atom: int
    c_atompair: int
    n_queries: int
    n_keys: int
    linear_no_bias_a: IndentationError
    layernorm_q: LayerNorm
    linear_no_bias_out: Linear
    atom_transformer: AtomTransformer

    def __call__(
        self,
        input_feature_dict: dict[str],
        a,
        q_skip,
        c_skip,
        p_skip,
    ):
        # Broadcast per-token activiations to per-atom activations and add the skip connection
        q = (
            broadcast_token_to_atom(
                x_token=self.linear_no_bias_a(a),  # [..., N_token, c_atom]
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            )  # [..., N_atom, c_atom]
            + q_skip
        )

        # Cross attention transformer
        q = self.atom_transformer(q, c_skip, p_skip)

        # Map to positions update
        return self.linear_no_bias_out(self.layernorm_q(q))


@register_from_torch(protenix.model.modules.diffusion.DiffusionModule)
class DiffusionModel(AbstractFromTorch):
    sigma_data: float
    c_atom: int
    c_atompair: int
    c_token: int
    c_s_inputs: int
    c_s: int
    c_z: int

    diffusion_conditioning: DiffusionConditioning
    atom_attention_encoder: AtomAttentionEncoder
    layernorm_s: LayerNorm
    linear_no_bias_s: Linear

    diffusion_transformer: DiffusionTransformer
    layernorm_a: LayerNorm

    atom_attention_decoder: AtomAttentionDecoder

    def f_forward(
        self,
        r_noisy,
        t_hat_noise_level,
        input_feature_dict: dict[str, jnp.ndarray],
        s_inputs,
        s_trunk,
        z_trunk,
        use_conditioning=False,
    ):
        N_sample = r_noisy.shape[-3]
        assert t_hat_noise_level.shape[-1] == N_sample
        # TODO: potentially checkpoint this
        s_single, z_pair = self.diffusion_conditioning(
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_conditioning=use_conditioning,
        )

        # Expand embeddings to match N_sample

        s_trunk = jnp.tile(s_trunk, (N_sample, 1, 1))
        z_pair = jnp.tile(z_pair, (N_sample, 1, 1, 1))
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            r_l=r_noisy,
            s=s_trunk,
            z=z_pair,
        )

        a_token = a_token.astype(jnp.float32)  # Ensure float32 for numerical stability
        a_token += self.linear_no_bias_s(self.layernorm_s(s_single))

        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
        )
        a_token = self.layernorm_a(a_token)

        return self.atom_attention_decoder(
            input_feature_dict=input_feature_dict,
            a=a_token,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
        )

    def __call__(
        self,
        x_noisy,
        t_hat_noise_level,
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        use_conditioning=True,
    ):
        """Forward pass of the diffusion model.

        Args:
            x_noisy: Noisy input coordinates.
            t_hat_noise_level: Noise level for the diffusion process.
            input_feature_dict: Dictionary containing input features.
            s_inputs: Single conditioning inputs.
            s_trunk: Trunk embeddings for single conditioning.
            z_trunk: Trunk embeddings for pair conditioning.
            use_conditioning: Whether to use conditioning in the forward pass.

        Returns:
            Output coordinates after processing through the model.
        """

        r_noisy = (
            x_noisy
            / jnp.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            use_conditioning=use_conditioning,
        )

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None]
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / jnp.sqrt(1 + s_ratio**2) * r_update
        )

        return x_denoised


@register_from_torch(protenix.model.generator.InferenceNoiseScheduler)
class InferenceNoiseScheduler(eqx.Module):
    sigma_data: float
    s_max: float
    s_min: float
    rho: float

    def __call__(self, N_step: int):
        step_size = 1 / N_step
        step_indices = jnp.arange(N_step + 1)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list = t_step_list.at[..., -1].set(0)  # t_N = 0

        return t_step_list

    @staticmethod
    def from_torch(m: protenix.model.generator.InferenceNoiseScheduler):
        return InferenceNoiseScheduler(
            sigma_data=m.sigma_data,
            s_max=m.s_max,
            s_min=m.s_min,
            rho=m.rho,
        )


def sample_diffusion(
    *,
    denoise_net,
    input_feature_dict: dict[str],
    s_inputs,
    s_trunk,
    z_trunk,
    noise_schedule,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    key,
):
    N_atom = input_feature_dict["atom_to_token_idx"].shape[-1]
    batch_shape = s_inputs.shape[:-2]


    # init noise
    # [..., N_sample, N_atom, 3]
    x_l = noise_schedule[0] * jax.random.normal(
        key=key, shape=(*batch_shape, N_sample, N_atom, 3)
    )

    
    
    def body_function(T, in_T):
        x_l, key = T
        c_tau_last, c_tau = in_T
        x_l = x_l - jnp.mean(x_l, axis=-2, keepdims=True)  # Center the coordinates

        # Denoise with a predictor-corrector sampler
        # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
        gamma = jax.lax.select(c_tau > gamma_min, gamma0, 0.0)
        t_hat = c_tau_last * (gamma + 1)

        delta_noise_level = jnp.sqrt(t_hat**2 - c_tau_last**2)
        key = jax.random.fold_in(key, 1)
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * jax.random.normal(
            key=key, shape=x_l.shape
        )

        # 2. Denoise from x_{t_hat} to x_{c_tau}
        # Euler step only
        t_hat = (
            jnp.tile(
                t_hat.reshape((1,) * (len(batch_shape) + 1)), (*batch_shape, N_sample)
            )  # [..., N_sample]
        )


        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
        )

        delta = (x_noisy - x_denoised) / t_hat[
            ..., None, None
        ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
        return (x_l, key), None

    x_l, key = jax.lax.scan(body_function,
        init=(x_l, key),
        xs=(noise_schedule[:-1], noise_schedule[1:]),
    )[0]

    

    
    return x_l


@register_from_torch(protenix.model.modules.pairformer.TemplateEmbedder)
class TemplateEmbedder(AbstractFromTorch):
    n_blocks: int
    c: int
    c_z: int
    input_feature1: dict
    input_feature2: dict
    distogram: dict
    inf: float
    linear_no_bias_z: Linear
    layernorm_z: LayerNorm
    linear_no_bias_a: Linear
    pairformer_stack: Pairformer
    layernorm_v: LayerNorm
    relu: any
    linear_no_bias_u: Linear

    def __call__(self, input_feature_dict, z, pair_mask=None, *, key):
        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            return jnp.zeros_like(z)

        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(z.dtype)

        num_residues = z.shape[0]
        num_templates = input_feature_dict["template_aatype"].shape[0]
        query_num_channels = z.shape[-1]

        if pair_mask is None:
            pair_mask = jnp.ones(z.shape[:-1])

        z_normed = self.layernorm_z(z)
        u = jnp.zeros_like(z_normed[..., :self.c])

        # Stack template features for scan: shapes are (N_templates, ...)
        template_xs = (
            input_feature_dict["template_distogram"],
            input_feature_dict["template_pseudo_beta_mask"],
            input_feature_dict["template_aatype"],
            input_feature_dict["template_unit_vector"],
            input_feature_dict["template_backbone_frame_mask"],
        )

        def body_fn(carry, xs):
            u, key = carry
            dgram, pseudo_beta_mask_2d, aatype, unit_vector, backbone_mask_2d = xs
            key = jax.random.fold_in(key, 1)

            # Apply masks
            dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
            pseudo_beta_mask_2d = pseudo_beta_mask_2d * multichain_mask * pair_mask

            # One-hot encode aatype
            aatype = jax.nn.one_hot(aatype, 32)  # (N_token, 32)

            # Apply masks to unit_vector and backbone_mask
            unit_vector = unit_vector * multichain_mask[..., None] * pair_mask[..., None]
            backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask

            # Concatenate all features
            to_concat = [
                dgram,
                pseudo_beta_mask_2d[..., None],
                # expand_at_dim(aatype, dim=-3): (1, N_token, 32) → (N_token, N_token, 32)
                jnp.broadcast_to(aatype[None, :, :], (z.shape[0], *aatype.shape)),
                # expand_at_dim(aatype, dim=-2): (N_token, 1, 32) → (N_token, N_token, 32)
                jnp.broadcast_to(aatype[:, None, :], (aatype.shape[0], z.shape[0], aatype.shape[-1])),
                unit_vector,
                backbone_mask_2d[..., None],
            ]
            at = jnp.concatenate(to_concat, axis=-1)
            v = self.linear_no_bias_z(z_normed) + self.linear_no_bias_a(at)
            _, v = self.pairformer_stack(s=None, z=v, pair_mask=pair_mask, key=key)
            v = self.layernorm_v(v)
            u = u + v
            return (u, key), None

        (u, _), _ = jax.lax.scan(body_fn, (u, key), template_xs)

        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(jax.nn.relu(u))
        return u


@register_from_torch(protenix.model.modules.confidence.ConfidenceHead)
class ConfidenceHead(AbstractFromTorch):
    b_pae: int
    b_pde: int
    b_plddt: int
    linear_no_bias_s1: Linear
    linear_no_bias_s2: Linear
    lower_bins: jax.Array
    upper_bins: jax.Array

    linear_no_bias_d: Linear
    linear_no_bias_d_wo_onehot: Linear

    pairformer_stack: Pairformer
    linear_no_bias_pae: Linear

    linear_no_bias_pde: Linear
    plddt_weight: jax.Array # shouldn't be part of the model, these people are crazy.
    resolved_weight: jax.Array

    input_strunk_ln: LayerNorm
    pae_ln: LayerNorm
    pde_ln: LayerNorm
    plddt_ln: LayerNorm
    resolved_ln: LayerNorm

    def __call__(self, *, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, key, use_embedding=True):
        s_trunk = self.input_strunk_ln(jnp.clip(s_trunk, -512, 512))#torch.clamp(s_trunk, min=-512, max=512))
        z_trunk = use_embedding * z_trunk

        x_rep_idx = input_feature_dict["atom_rep_atom_idx"]
        x_pred_rep_coords = x_pred_coords[..., x_rep_idx, :]
        N_sample = x_pred_rep_coords.shape[-3]

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk

        def single_structure(x_pred_rep_coords, key):
            z_pair = z_trunk

            distance_pred = jnp.sqrt(
                ((x_pred_rep_coords[..., None, :] - x_pred_rep_coords[..., None, :, :]) ** 2 + 1E-9).sum( axis=-1)
            )  # [..., N_tokens, N_tokens]
            z_pair += self.linear_no_bias_d(
                ((distance_pred[..., None] > self.lower_bins) * (distance_pred[..., None] < self.upper_bins)).astype(jnp.float32)
            )  # [..., N_tokens, N_tokens, c_z]
            z_pair += self.linear_no_bias_d_wo_onehot(
                distance_pred[..., None],
            )  # [..., N_tokens, N_tokens, c

            s_single, z_pair = self.pairformer_stack(
                s=s_trunk,
                z=z_pair,
                pair_mask=pair_mask,
                key=key)
            
            atom_to_token_idx = input_feature_dict[
            "atom_to_token_idx"
            ]  # in range [0, N_token-1] shape: [N_atom]
            atom_to_tokatom_idx = input_feature_dict[
                "atom_to_tokatom_idx"
            ] 
            pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
            pde_pred = self.linear_no_bias_pde(
                self.pde_ln(z_pair + jnp.swapaxes(z_pair, -2, -3))
            )
            # Broadcast s_single: [N_tokens, c_s] -> [N_atoms, c_s]
            a = broadcast_token_to_atom(
                x_token=s_single, atom_to_token_idx=atom_to_token_idx
            )
            plddt_pred = jnp.einsum(
                "...nc,ncb->...nb",
                self.plddt_ln(a),
                self.plddt_weight[atom_to_tokatom_idx],
            )
            resolved_pred = jnp.einsum(
                "...nc,ncb->...nb",
                self.resolved_ln(a),
                self.resolved_weight[atom_to_tokatom_idx],
            )
            return plddt_pred, pae_pred, pde_pred, resolved_pred
        
        plddt_pred, pae_pred, pde_pred, resolved_pred = jax.vmap(single_structure)(x_pred_rep_coords, jax.random.split(key, N_sample))

        return plddt_pred, pae_pred, pde_pred, resolved_pred



class InitialEmbedding(eqx.Module):
    s_init: Float[Array, "...  N_token c_s"]
    z_init: Float[Array, "... N_token N_token c_z"]
    s_inputs: Float[Array, "... N_token c_s_inputs"]


class TrunkEmbedding(eqx.Module):
    s: Float[Array, "... N_token c_s"]
    z: Float[Array, "... N_token N_token c_z"]

class ConfidenceMetrics(eqx.Module):
    plddt_logits: Float[Array, "... N_sample N_token 50"]
    pae_logits: Float[Array, "... N_sample N_token N_token 64"]
    pde_logits: Float[Array, "... N_sample N_token N_token 64"]
    resolved_logits: Float[Array, "... N_sample N_token 2"]


class Outputs(eqx.Module):
    coordinates: Float[Array, "... N_sample N_atom 3"]
    confidence_metrics: ConfidenceMetrics
    distogram_logits: Float[Array, "... N_sample N_token N_token 64"]

@register_from_torch(protenix.model.protenix.Protenix)
class Protenix(eqx.Module):
    input_embedder: InputFeatureEmbedder
    relative_position_encoding: RelativePositionEncoding
    template_embedder: TemplateEmbedder
    msa_module: MSAModule
    pairformer_stack: Pairformer
    distogram_head: DistogramHead
    linear_no_bias_sinit: Linear
    linear_no_bias_zinit1: Linear
    linear_no_bias_zinit2: Linear
    linear_no_bias_token_bond: Linear
    linear_no_bias_z_cycle: Linear
    linear_no_bias_s: Linear
    layernorm_z_cycle: LayerNorm
    layernorm_s: LayerNorm
    diffusion_module: DiffusionModel
    inference_noise_scheduler: InferenceNoiseScheduler
    N_steps: int
    gamma0: float
    gamma_min: float
    noise_scale_lambda: float
    step_scale_eta: float
    confidence_head: ConfidenceHead

    @staticmethod
    def from_torch(m: protenix.model.protenix.Protenix):
        return Protenix(
            input_embedder=from_torch(m.input_embedder),
            relative_position_encoding=from_torch(m.relative_position_encoding),
            template_embedder=from_torch(m.template_embedder),
            msa_module=from_torch(m.msa_module),
            pairformer_stack=from_torch(m.pairformer_stack),
            distogram_head=from_torch(m.distogram_head),
            linear_no_bias_sinit=from_torch(m.linear_no_bias_sinit),
            linear_no_bias_zinit1=from_torch(m.linear_no_bias_zinit1),
            linear_no_bias_zinit2=from_torch(m.linear_no_bias_zinit2),
            linear_no_bias_token_bond=from_torch(m.linear_no_bias_token_bond),
            linear_no_bias_z_cycle=from_torch(m.linear_no_bias_z_cycle),
            linear_no_bias_s=from_torch(m.linear_no_bias_s),
            layernorm_z_cycle=from_torch(m.layernorm_z_cycle),
            layernorm_s=from_torch(m.layernorm_s),
            diffusion_module=from_torch(m.diffusion_module),
            inference_noise_scheduler=from_torch(m.inference_noise_scheduler),
            N_steps=m.configs.sample_diffusion["N_step"],
            gamma0=m.configs.sample_diffusion["gamma0"],
            gamma_min=m.configs.sample_diffusion["gamma_min"],
            noise_scale_lambda=m.configs.sample_diffusion["noise_scale_lambda"],
            step_scale_eta=m.configs.sample_diffusion["step_scale_eta"],
            confidence_head=from_torch(m.confidence_head),
        )

    @eqx.filter_jit
    def embed_inputs(self, *, input_feature_dict) -> InitialEmbedding:
        s_inputs = self.input_embedder(input_feature_dict)
        s_init = self.linear_no_bias_sinit(s_inputs)  #  [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]

        z_init += self.relative_position_encoding(input_feature_dict)
        z_init += self.linear_no_bias_token_bond(
            input_feature_dict["token_bonds"][..., None]
        )
        return InitialEmbedding(
            s_init=s_init,
            z_init=z_init,
            s_inputs=s_inputs,
        )
    
    @eqx.filter_jit
    def recycle(self, *, initial_embedding: InitialEmbedding, input_feature_dict, recycling_steps: int, key, state = None):
        if state is None:
            state = TrunkEmbedding(
                s=jnp.zeros_like(initial_embedding.s_init),
                z=jnp.zeros_like(initial_embedding.z_init),
            )

        def body_fn(state: TrunkEmbedding, key):
            state = jax.lax.stop_gradient(state)  # Prevent gradient flow through recycling
            s,z = state.s, state.z
            z = initial_embedding.z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            if self.template_embedder.n_blocks > 0:
                z = z + self.template_embedder(
                    input_feature_dict, z, pair_mask=None, key=key
                )
            z = self.msa_module(
                input_feature_dict, z, initial_embedding.s_inputs, pair_mask=None, key=key
            )
            s = initial_embedding.s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s, z = self.pairformer_stack(s, z, pair_mask=None, key=jax.random.fold_in(key, 1))
            return TrunkEmbedding(s=s, z=z), None
        
        state, _ = jax.lax.scan(
            body_fn,
            init=state,
            xs=jax.random.split(key, recycling_steps),
        )

        return state


    @eqx.filter_jit
    def sample_structures(self, *, initial_embedding: InitialEmbedding, trunk_embedding: TrunkEmbedding, input_feature_dict, N_samples, N_steps, key):
        noise_schedule = self.inference_noise_scheduler(N_step=N_steps)
        coordinates = sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=initial_embedding.s_inputs,
            s_trunk=trunk_embedding.s,
            z_trunk=trunk_embedding.z,
            N_sample=N_samples,
            noise_schedule=noise_schedule,
            gamma0=self.gamma0,
            gamma_min=self.gamma_min,
            noise_scale_lambda=self.noise_scale_lambda,
            step_scale_eta=self.step_scale_eta,
            key=key,
        )
        return coordinates
    
    @eqx.filter_jit
    def confidence_metrics(self, *, initial_embedding: InitialEmbedding, trunk_embedding: TrunkEmbedding, input_feature_dict, coordinates, key): 
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=initial_embedding.s_inputs,
            s_trunk=trunk_embedding.s,
            z_trunk=trunk_embedding.z,
            pair_mask=None,
            x_pred_coords=coordinates,
            key=key,
            use_embedding=True,
        )

        return ConfidenceMetrics(
            plddt_logits=plddt_pred,
            pae_logits=pae_pred,
            pde_logits=pde_pred,
            resolved_logits=resolved_pred,
        )

        

    def __call__(self, *, input_feature_dict, N_cycle, N_sample, N_steps=None, key):
        if N_steps is None:
            N_steps = self.N_steps

        initial_embedding = self.embed_inputs(input_feature_dict=input_feature_dict)
        # Recycling iterations
        trunk_embedding = self.recycle(
            initial_embedding=initial_embedding,
            input_feature_dict=input_feature_dict,
            recycling_steps=N_cycle,
            key=key,
        )
        # Sample structures
        coordinates = self.sample_structures(
            initial_embedding=initial_embedding,
            trunk_embedding=trunk_embedding,
            input_feature_dict=input_feature_dict,
            N_samples=N_sample,
            N_steps=N_steps,
            key=key,
        )
        # Confidence metrics
        confidence_metrics = self.confidence_metrics(
            initial_embedding=initial_embedding,
            trunk_embedding=trunk_embedding,
            input_feature_dict=input_feature_dict,
            coordinates=coordinates,
            key=key,
        )

        return Outputs(
            coordinates=coordinates,
            confidence_metrics=confidence_metrics,
            distogram_logits=self.distogram_head(trunk_embedding.z),
        )
        