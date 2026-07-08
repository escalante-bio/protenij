"""Forward+backward benchmark: vanilla vs cueq-fast protenij.

Uses an MSE-on-distogram-logits loss that exercises the full trunk
(input embedder, MSA module, template embedder, Pairformer stack, distogram
head). Diffusion + confidence head are excluded — they host no cueq-swapped
kernels and the sampler adds stochasticity that complicates grad timing.

This is the relevant regime for training: cueq's cuda-fwd/cuda-bwd kernels for
triangle_attention (and triton VJP for triangle_multiplicative_update) get
exercised end-to-end.
"""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import argparse
import statistics
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from protenix.backend import load_model

from bench_folding import SEQUENCES, enable_cueq, build_features


def trunk_loss(model, features, key, n_cycle=4):
    """MSE of distogram logits against a zero target. Scalar — grad OK.

    recycle() stop_gradients between cycles, so only the last cycle contributes
    to grad. We still run n_cycle forwards to match production wall time.
    """
    emb = model.embed_inputs(input_feature_dict=features)
    trunk = model.recycle(
        initial_embedding=emb, input_feature_dict=features,
        recycling_steps=n_cycle, key=key,
    )
    logits = model.distogram_head(trunk.z)
    return jnp.mean(jnp.square(logits))


def _block(tree):
    jax.tree.map(
        lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else None,
        tree,
    )


def _time(fn, *, n_iter):
    # First call includes compile; time it separately.
    t0 = time.perf_counter()
    out = fn()
    _block(out)
    compile_plus_run = time.perf_counter() - t0

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = fn()
        _block(out)
        times.append(time.perf_counter() - t0)
    return compile_plus_run, times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="protenix_base_default_v1.0.0")
    parser.add_argument("--n_cycle", type=int, default=4)
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--msa_dir", default="output_test_predict/msa",
                        help="mmseqs2 MSA cache dir; set to '' for dummy MSA")
    args = parser.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"loading {args.model}...")
    baseline = load_model(args.model)
    cueq = enable_cueq(baseline)

    msa_dir = args.msa_dir if args.msa_dir else None
    n_cycle = args.n_cycle

    def _fwd(model, features, key):
        return trunk_loss(model, features, key, n_cycle=n_cycle)

    fwd = eqx.filter_jit(_fwd)
    fwd_bwd = eqx.filter_jit(eqx.filter_value_and_grad(_fwd))

    print(f"\nn_cycle={n_cycle}  n_iter={args.n_iter}  "
          f"msa={'real (mmseqs2)' if msa_dir else 'dummy'}")
    print("loss = mean(distogram_logits ** 2)")

    for name, seq in SEQUENCES.items():
        features = build_features(seq, name, msa_dir=msa_dir)
        n_tok = int(features["token_index"].shape[0])
        print(f"\n=== {name} ({len(seq)} residues, {n_tok} tokens, "
              f"msa={tuple(features['msa'].shape)}) ===")
        key = jax.random.PRNGKey(0)

        results = {}
        for label, model in [("baseline", baseline), ("cueq-fast", cueq)]:
            c_fwd, t_fwd = _time(lambda: fwd(model, features, key), n_iter=args.n_iter)
            c_bb,  t_bb  = _time(lambda: fwd_bwd(model, features, key), n_iter=args.n_iter)
            mf = statistics.mean(t_fwd) * 1e3
            mb = statistics.mean(t_bb)  * 1e3
            results[label] = (mf, mb, mb - mf)
            print(f"  {label:10s}  "
                  f"fwd mean={mf:6.1f}ms (compile+1st={c_fwd:6.2f}s)   "
                  f"fwd+bwd mean={mb:6.1f}ms (compile+1st={c_bb:6.2f}s)   "
                  f"bwd≈{mb - mf:6.1f}ms")

        mf_b, mfb_b, bwd_b = results["baseline"]
        mf_c, mfb_c, bwd_c = results["cueq-fast"]
        print(f"  speedup (cueq vs baseline): "
              f"fwd={mf_b/mf_c:.2f}x   bwd≈{bwd_b/bwd_c:.2f}x   "
              f"fwd+bwd={mfb_b/mfb_c:.2f}x")


if __name__ == "__main__":
    main()
