# Atom-padding work log

- Isolated branch from d48a0b0; source checkout, running evaluations, dependency pins and existing caches untouched.
- Audited `ref_mask`: reference-conformer availability, not atom presence. Added separate `atom_pad_mask`.
- Implemented optional host feature padding (configurable pitch, default 256 when explicitly requested), masked local atom attention/pooling/diffusion, safe atom indexing and masked confidence outputs, and host output unpadding.
- Kept model parameter dataclasses unchanged for serialized-checkpoint compatibility; no-mask calls retain original behavior/RNG calls.
- Eight checkpoint-free CPU regression tests passed, covering adversarial NaN padding, local attention boundary/empty windows and multiple samples, diffusion parity/default RNG agreement, output exclusion and one JIT trace across native counts.
- Real Protenix-v2 checkpoint on an 8-token/59-atom fragment, padded to 256: input/trunk embeddings bit exact; five-step controlled-noise coordinate RMS difference 0.000159 Å (maximum 0.000587 Å), confidence-logit differences ≤1.63e-5 on CPU. Also passed with invalid padded metadata/indices and NaN padded noise.
- Two actual 515-token complex fixtures with 4182 and 4178 native atoms have identical full feature signatures at 4352 atoms. Private fixtures are not committed.
- H100 validation, highest matmul precision, one recycle/five steps/one sample: native-versus-padded input/trunk embeddings and distograms bit exact; coordinate RMS differences 0.000374/0.000359 Å, maxima 0.00556/0.00498 Å. Confidence logits pass the 1e-3 absolute/relative check; coordinates exceed this strict elementwise threshold. The harness records this failure rather than hiding it.
- Isolated full-complex fixed-input denoiser comparison is bit exact. Initial centering differs by maximum 3.05e-5 Å, RMS 5.31e-6 Å when its reduction extent changes; fixed-bucket NaN padding changes centering by exactly zero. Small floating-point centering differences propagate through the iterative sampler.
- First padded full-complex lowering/compile/warm execution: 0.869/26.664/5.179 seconds. Second padded sequence: 0.0061/0.00027/5.181 seconds and zero new traces. Second native shape instead compiled for 23.864 seconds. The first native cold compilation took 110.384 seconds; these are short-harness timings, not production evaluation estimates.
- Preallocation stayed enabled with a 65% GPU budget. Validation processes finished and released the GPU. Production 10-recycle/200-step equivalence and downstream adapter integration are not claimed by this draft.
