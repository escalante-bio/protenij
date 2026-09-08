# Atom-padding work log

- Isolated branch from d48a0b0; source checkout, running evaluations and existing caches untouched.
- Audited `ref_mask`: reference-conformer availability, not atom presence. Added separate `atom_pad_mask`.
- Implemented optional host feature padding (configurable pitch, default 256 when explicitly requested), masked local atom attention/pooling/diffusion, safe atom indexing and masked confidence outputs, and host output unpadding.
- Kept model parameter dataclasses unchanged for serialized-checkpoint compatibility; no-mask calls retain original behavior.
- Added controlled-noise hooks to the existing sampler for parity testing, without changing default RNG calls.
- Eight checkpoint-free CPU regression tests passed, including adversarial NaN padding, local attention boundary/empty windows and multiple samples, diffusion parity, output exclusion and one JIT trace for multiple native counts.
- Two actual 515-token VHH-complex fixtures with 4182 and 4178 native atoms have identical full feature signatures after padding to 4352 atoms. Private fixtures are not committed.
- Real Protenix-v2 checkpoint on an 8-token/59-atom fragment, padded to 256: input/trunk embeddings bit exact; five-step controlled-noise coordinate RMS difference 0.000159 Å (maximum 0.000587 Å), confidence-logit differences ≤1.63e-5 on CPU.
- Full 515-token GPU parity and compilation/warm timing validation pending shared GPU availability.
