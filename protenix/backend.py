# See `joltz.py` for an introduction.

from dataclasses import fields
from functools import singledispatch
import os
import pickle

import time
import einops
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from functools import partial
import numpy as np
from jax import tree

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── Equinox module definitions (always available, no torch needed) ───────────


class relu(eqx.Module):
    def __call__(self, x):
        return jax.nn.relu(x)

class sigmoid(eqx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)

class silu(eqx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)


class softmax(eqx.Module):
    dim: int

    def __call__(self, x):
        return jax.nn.softmax(x, axis=self.dim)

class Identity(eqx.Module):
    def __call__(self, x):
        return x

# this isn't very jax-y
def _vmap(f, tensor, *args):
    for _ in range(len(tensor.shape) - 1):
        f = jax.vmap(f)
    return f(tensor, *args)


def vmap_to_last_dimension(f):
    return partial(_vmap, f)


class Linear(eqx.Module):
    """Linear layer that matches pytorch semantics"""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = einops.einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o


class LayerNorm(eqx.Module):
    """LayerNorm that matches pytorch semantics"""

    weight: Float[Array, "Out"] | None
    bias: Float[Array, "Out"] | None
    eps: float

    def __call__(self, x: Float[Array, "... Out"]) -> Float[Array, "... Out"]:
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class Sequential(eqx.Module):
    _modules: dict[
        str, eqx.Module
    ]  # IMHO this is a fairly wild design choice, but this is really how pytorch works.

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x


class SparseEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __call__(self, indices):
        ndims = len(indices.shape)

        def apply(index):
            return self.embedding(index)

        f = apply
        for _ in range(ndims):
            f = jax.vmap(f)

        return f(indices)


# ── Serialization (torch-free) ──────────────────────────────────────────────


def save_model(model, path):
    """Save an Equinox model to disk as .eqx (arrays) + .skeleton.pkl (structure)."""
    eqx.tree_serialise_leaves(f"{path}.eqx", model)
    skeleton = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x,
        model,
        is_leaf=eqx.is_array,
    )
    with open(f"{path}.skeleton.pkl", "wb") as f:
        pickle.dump(skeleton, f, protocol=pickle.HIGHEST_PROTOCOL)


HF_REPO = "nickrb/protenij"

KNOWN_MODELS = [
    "protenix_tiny_default_v0.5.0",
    "protenix_mini_default_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
]

DATA_FILES = [
    "components.v20240608.cif",
    "components.v20240608.cif.rdkit_mol.pkl",
    "clusters-by-entity-40.txt",
]

_CACHE_DIR = os.path.expanduser("~/.protenix")


def _hf_download(filename: str) -> None:
    from huggingface_hub import hf_hub_download

    os.makedirs(_CACHE_DIR, exist_ok=True)
    hf_hub_download(HF_REPO, filename, local_dir=_CACHE_DIR)


def download_data() -> None:
    """Download CCD and PDB cluster data files from HuggingFace if missing."""
    for name in DATA_FILES:
        if not os.path.exists(os.path.join(_CACHE_DIR, name)):
            _hf_download(name)


def _resolve_model_path(name_or_path: str) -> str:
    """Resolve a model name to a local path, downloading from HF if needed."""
    path = os.path.expanduser(name_or_path)
    if os.path.exists(f"{path}.eqx"):
        return path

    # Check ~/.protenix/ cache
    cache_path = os.path.join(_CACHE_DIR, name_or_path)
    if os.path.exists(f"{cache_path}.eqx"):
        return cache_path

    # Download from HuggingFace
    for suffix in [".eqx", ".skeleton.pkl"]:
        _hf_download(f"{name_or_path}{suffix}")
    return cache_path


def load_model(name_or_path: str):
    """Load an Equinox model (no PyTorch needed).

    Args:
        name_or_path: Either a full path (e.g. '~/.protenix/protenix_base_default_v1.0.0')
            or a model name (e.g. 'protenix_base_default_v1.0.0') which will be
            resolved from ~/.protenix/ or downloaded from HuggingFace.
    """
    import protenix.protenij  # ensure pytree node types are registered

    download_data()
    path = _resolve_model_path(name_or_path)
    with open(f"{path}.skeleton.pkl", "rb") as f:
        skeleton = pickle.load(f)
    return eqx.tree_deserialise_leaves(f"{path}.eqx", skeleton)


# ── Torch-dependent conversion utilities (only available when torch installed) ──


if _HAS_TORCH:

    @singledispatch
    def from_torch(x):
        raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")

    # basic types
    from_torch.register(torch.Tensor, lambda x: np.array(x.cpu().detach()))
    from_torch.register(int, lambda x: x)
    from_torch.register(float, lambda x: x)
    from_torch.register(bool, lambda x: x)
    from_torch.register(str, lambda x: x)
    from_torch.register(type(None), lambda x: x)
    from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
    from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})
    from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])

    class AbstractFromTorch(eqx.Module):
        """
        Default implementation of `from_torch` for equinox modules.
        This checks that the fields of the equinox module are present in the torch module and constructs the equinox module from the torch module by recursively calling `from_torch` on the children of the torch module.
        Allows for missing fields in the torch module if the corresponding field in the equinox module is optional.

        """

        @classmethod
        def from_torch(cls, model: torch.nn.Module):
            # assemble arguments to `cls` constructor from `model`

            field_to_type = {field.name: field.type for field in fields(cls)}
            kwargs = {
                child: from_torch(child_module)
                for child, child_module in model.named_children()
            } | {
                parameter_name: from_torch(parameter)
                for parameter_name, parameter in model.named_parameters(recurse=False)
            }

            # add fields that are not child_modules or parameters
            for field_name, field_type in field_to_type.items():
                if not hasattr(model, field_name):
                    if not isinstance(None, field_type):
                        raise ValueError(
                            f"Field {field_name} for {cls} is not optional but is missing from torch model {model}"
                        )
                    else:
                        kwargs[field_name] = None
                else:
                    kwargs[field_name] = from_torch(getattr(model, field_name))

            # check we're not passing any additional properties
            torch_not_equinox = kwargs.keys() - field_to_type.keys()
            if torch_not_equinox:
                raise ValueError(
                    f"Properties in torch model not found in equinox module {cls}: {torch_not_equinox}"
                )

            return cls(**kwargs)

    def register_from_torch(torch_module_type):
        """Class decorator to register an equinox module for conversion from a torch module."""

        def decorator(cls):
            from_torch.register(torch_module_type, cls.from_torch)
            return cls

        return decorator

    # Register from_torch converters for all Equinox modules
    from_torch.register(torch.nn.ReLU, lambda _: relu())
    from_torch.register(torch.nn.Sigmoid, lambda _: sigmoid())
    from_torch.register(torch.nn.SiLU, lambda _: silu())
    from_torch.register(torch.nn.Softmax, lambda m: softmax(dim=m.dim))
    from_torch.register(torch.nn.Identity, lambda _: Identity())

    @staticmethod
    def _linear_from_torch(l: torch.nn.Linear):
        return Linear(weight=from_torch(l.weight), bias=from_torch(l.bias))
    from_torch.register(torch.nn.Linear, _linear_from_torch)

    @staticmethod
    def _layernorm_from_torch(l: torch.nn.LayerNorm):
        return LayerNorm(
            weight=from_torch(l.weight), bias=from_torch(l.bias), eps=l.eps
        )
    from_torch.register(torch.nn.LayerNorm, _layernorm_from_torch)

    @staticmethod
    def _sequential_from_torch(module: torch.nn.Sequential):
        return Sequential(_modules=from_torch(module._modules))
    from_torch.register(torch.nn.Sequential, _sequential_from_torch)

    @staticmethod
    def _embedding_from_torch(m: torch.nn.modules.sparse.Embedding):
        return SparseEmbedding(embedding=eqx.nn.Embedding(weight=from_torch(m.weight)))
    from_torch.register(torch.nn.modules.sparse.Embedding, _embedding_from_torch)

    # Useful for testing
    class TestModule(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.mod = module
            self.j_m = eqx.filter_jit(from_torch(self.mod))

        def forward(self, *args, **kwargs):
            torch_start = time.time()
            torch_output = self.mod(*args, **kwargs)
            torch_end = time.time()

            jax_start = time.time()
            with jax.default_matmul_precision("float32"):
                jax_output = self.j_m(*from_torch(args), **from_torch(kwargs))
            tree.map(lambda v: v.block_until_ready(), jax_output)
            jax_end = time.time()

            errors = tree.map(
                lambda a, b: jnp.abs(jnp.array(a) - b).max()
                if isinstance(b, jnp.ndarray)
                else None,
                torch_output,
                jax_output,
                is_leaf=eqx.is_inexact_array,
            )
            print(f"max abs error {type(self.mod)}: ", errors)
            print(
                f"torch time: {torch_end - torch_start : .3f}s, jax time: {jax_end - jax_start : .3f}s"
            )
            return torch_output

else:
    # When torch is not available, provide stub base class so protenij.py
    # class definitions still work (they extend AbstractFromTorch).
    AbstractFromTorch = eqx.Module
