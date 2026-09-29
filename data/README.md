# Data

## Controlled burst workloads

Generated deterministically from a seed by `dsorch.workload.controlled_trace`.
Nothing needs to be downloaded.

## Public B5G trace replay (optional, `configs/b5g.yaml`)

The replay uses the processed slice-level trace `public_b5g_trace.csv`, which
the GAN-JCSO authors derived from the public network-slicing dataset of
Farreras et al. (Data in Brief 55:110738, 2024; raw data
doi:10.5281/zenodo.10610616). It is distributed with the supplementary material
of Qiu and Zhang (J. Cloud Comput., 2026, doi:10.1186/s13677-026-00985-4).

Because that supplementary file carries its own licence, it is **not**
committed here. To run the replay:

1. Download the GAN-JCSO supplementary material (Additional file 1).
2. Copy `public_b5g_trace.csv` into `data/external/`.

Expected columns include `seed, slot, slice, bw_demand, cpu_demand`; the file
holds three seed-selected 260-slot segments (seeds 3, 11 and 27) with the
FlowPathQoS CPU proxy. The label `mIoT` is mapped to `mMTC` at load time.
