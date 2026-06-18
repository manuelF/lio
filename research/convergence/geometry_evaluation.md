Your premise needs correcting. The data on this build doesn't support "we are still dominated [on the Fortran side]." With LIO_OVERLAP_INT3LU_G2G=1 and CUDA on (RTX 3080
Ti, 5 CPU + 1 GPU), per-iter steady state on fosfato is: 
 
g2g = 18.2msint3lu = 15.0msidle = 3.2ms (int3lu waits for g2g) 
 
g2g is the long pole by ~3ms. Wall is 1.71s, 26 SCF iters. Partition: 348 groups → 292 CPU (cost 5.1e8) + 56 GPU (cost 1.28e10), SPLIT_COST auto = 8.4M. 
 
The 3ms-per-iter ceiling. If you cut g2g from 18→15ms you save 78ms wall (~5%). Below 15ms int3lu becomes the long pole. So ideas split into three classes:
 
- (a) Recover the 3ms headroom — eat the idle directly. Cap ~80ms wall.
- (b) Push past 15ms by reducing total work (fewer points, fewer evals) — both sides win.
- (c) Forces phase (5.94s = 38% of wall, mostly QM/MM gradients). Separate frontier; partition barely touches it. Mentioned only where relevant. 
 
The single biggest signal is Pareto skew (from CPU-only perfmodel dump, 348 groups): 
 
top 1 group→ 105 ms (P=3834,M=258) 7% of total CPU work
top 5 groups → 308 ms21% 
top10 groups → 501 ms35% 
top25 groups → 933 ms64% 
top50 groups → 1342 ms 93% 
top 100 groups → 1442 ms 99.6% 
remaining 248 groups → 6 ms0.4%
 
137 groups (39%) consume <10 µs each. 1 group is a 105ms cliff. Both ends are wrong: tiny groups pay γ_overhead with no compute payoff, and the cliff is irreducibly serial
 on one CPU thread (or one GPU group on the GPU stream). The current cube-grid + per-atom-sphere geometry creates both pathologies by construction — uniform spatial cells 
in a non-uniform problem.
 
The auto-tuner already sweeps cube_size × sphere_radius × sphere_decomp. Ideas have to be structurally different to count, not parameter retuning. 
 
---
13 ideas, classified 
 
Geometry — beyond cubes+spheres
 
1. Adaptive octree on a target group cost. Replace the uniform cube grid with recursive 8-way subdivision. Stop subdividing when a leaf cell's predicted γ + α·P·M² falls
inside [T_target/2, T_target] (e.g. T_target ≈ γ_gpu × 5 = ~150 µs on this GPU). Tiny-region cells naturally merge with neighbours in coarser ancestors; dense regions 
auto-subdivide. Eliminates both the 137-tiny-group tail and most cliffs in one structural change.
Class (a). Target: −2 to −3 ms/iter, reach ceiling. Risk: cell-boundary M jumps (basis functions touch siblings) — must verify M doesn't double for small leaves.
 
2. KD-tree aligned to molecular PCA. Compute principal axes of the atom-position matrix, build an axis-aligned bounding box in PCA space, then recursively bisect on the 
axis with highest variance until each leaf has a target P. Anisotropic systems (peptides, DNA, the elongated fosfato) currently waste half their cube volume.
Class (a/b). Target: −5–10 % point count by avoiding empty cells (~b), better M-locality per cell (~a).
 
3. Voronoi/nearest-atom partitioning. Drop the cube/sphere dichotomy entirely. Every grid point is assigned to its nearest atom. This is the natural shape for 
atom-centered basis sets — M is roughly constant within an atom's basin and changes monotonically with distance. Splits like the current sphere geometry but extended to 
all points; sphere_decomp_size becomes the only knob.
Class (a/b). Target: more uniform M per group → tighter LPT. Risk: large basins for diffuse atoms still need internal subdivision. 

4. Function-cluster partitioning. Invert the loop. Cluster basis functions by their support (e.g. agglomerative on atom-pair distance), then for each function cluster,
gather the points where those functions are significant. Within a cluster M is constant by construction — RMM submatrix is reused, cache hit rate up, indexing simplifies.
The current code computes M after picking points, which is why M is so jagged across groups. 
Class (a/b). Target: M-uniform → P·M² fitting becomes near-perfect, both auto-tuner and bin-packer get more accurate, plus cache wins. 
 
5. Hilbert-curve point ordering, then equal-cost slicing. Sort all points along a 3-D Hilbert curve, scan once, cut whenever cumulative α·P·M² crosses T_target. Yields
spatially-coherent groups of bounded cost without any tree structure. Trivial to implement; deterministic. 
Class (a). Target: similar to octree but no recursion. Risk: still need to recompute M per cut.
 
Fusion & splitting — Pareto attack 
 
6. Tail fusion: merge tiny adjacent groups into super-groups, capped at γ-budget. After classifying functions, walk groups sorted by (M_signature, position) and fuse any
pair whose union has T_predicted ≤ T_target AND union(M) ≤ 1.2·max(M_a,M_b) (so we don't blow up M and amplify cost). Reduces 348→~100 groups, eliminating 248 × 
γ_overhead. With γ_gpu ≈ 28 µs per group on RTX 3080 Ti this is up to 7 ms of overhead on GPU side alone.
Class (a). Target: −2–4 ms/iter. Risk: non-trivial union-find on M sets; first iter no perf data → use cheap heuristic on assign_functions output. 
 
7. Head decapitation: split groups whose predicted time > median × N. Specifically, any group with T_pred > T_gpu_total/N_gpu_streams (currently 1) gets bisected on its 
longest spatial axis. The 105 ms cliff group (P=3834, M=258 in CPU perfmodel) becomes two halves of ~50 ms; it can now be parallelised between GPU and another worker, OR
distributed across multiple CUDA streams. Already done partially by sphere_decomp_size — extend the same logic to cubes and to a cube_decomp_size based on time, not point 
count. 
Class (a). Target: −1 ms/iter today. Larger payoff with idea 13 (multi-stream GPU).
 
8. Cube-sphere fusion at boundary. Currently cubes and spheres are built separately even when a sphere atom's outermost shell is well inside a neighbouring cube. The
Class (a). Small win, but cleans up the geometric model. 
 
Cost-model fixes 
 
9. γ-aware LPT bin packing. PointGroup::cost() is 10·P·M·(1+M)/2 + MINCOST — flat across devices, MINCOST=250000 is a kludge for "small group penalty."
cost_calibration.{h,cpp} already fits real α_cpu, α_gpu, γ_gpu from timeforgroup[]. Plumb those numbers into compute_work_partition() so LPT balances predicted finish time
 per thread, not cost units. Today the bin packer can land two 100 ms groups on the same thread because they have similar cost(); with a γ-aware model it would instead
spread them and accept slightly higher count imbalance.
Class (a). Target: −0.5–1 ms/iter (CPU max already 8 ms; with perfect pack it's ~6.7 ms). Cheapest-to-implement of all 13. 
 
10. Repartition on detected drift, not every SCF. Today regenerate_partition is 195 ms (cold) and re-runs every SCF entry. In MD step N+1 atoms move <0.1 Å. Cache the 
partition keyed on a hash of (atom_positions, basis_set, sphere_radius, cube_size); only invalidate cells whose atom contents changed (atoms enter/leave the cell or cross 
significance boundaries). On warm step expect <5 ms instead of 195 ms. 
Class (b/c). Target: −190 ms cold, recovered every MD step. In MD this is the single biggest savings. Risk: detecting "significance change" without full recompute requires
 storing per-atom radii. 
 
Heterogeneous within-group 
 
11. Per-group CPU/GPU split (within ONE group). Drop the binary CPU-vs-GPU device decision per group. For the largest groups, send a P-axis prefix to CPU and the suffix to
 GPU concurrently, then accumulate. move_base_from() already moves point/function buffers cheaply; with the new async pool (memory: cudaMallocAsync landed 2026-04-24) 
per-group device handoff is feasible. Useful when the cliff group is too big for the GPU stream alone in the per-iter budget.
Class (a). Target: −2–3 ms on largest groups. Risk: split-group reduction needs a synchroniser between devices.
 
12. Persistent CUDA kernel with device-side work queue. The 56 GPU groups today launch 56 kernels (×n stages) per iter. Replace with one persistent grid that pulls
(group_id, stage) tuples from a device-side queue. Eliminates 56 × kernel-launch latency (~few µs each) plus the cudaStreamSynchronize between groups (where applicable).
Different from the previously-rejected "multi-stream pipeline" — this is a single-stream single-kernel megakernel design.
Class (a). Target: −1–2 ms/iter. Risk: register pressure if the megakernel covers all stages; may need to split into 2 (functions+density vs rmm). 
13. Density-aware point pruning, recomputed each iter. After SCF iter 1, mark points where ρ(r) < ε (for an ε chosen from the integration error budget). Skip them in
subsequent iters: no φ evaluation, no rmm contribution. Typical pruning rate in NWChem-style codes is 30–50 % of points after iter 2 for organics. This reduces both sides 
— fewer Fock contributions for int3lu's QM/MM coupling region and fewer XC integrations for g2g. Single biggest class-(b) lever in this list.
*Class (b). Target: −20–40 % per-iter on both g2g and int3lu. Risk: pruning shifts XC integral, which can poison DIIS history (same hazard that killed XC-grid scheduling
2026-04-25 — you have prior data on the failure mode). Mitigation: prune only after DIIS converges past damping_off, treat as a finishing pass.* 
 
---
What I'd start with
 
Cheap, high signal-to-noise, no algorithmic risk, in this order: 
1. #9 (γ-aware bin pack) — half-day change, immediate measurable.
2. #6 (tail fusion) — 1-2 day change, addresses the 137-tiny-group tail. 
3. #10 (incremental MD repartition) — only matters if you ship MD; biggest absolute savings outside SCF. 
4. Then prototype #1 or #3 (geometry) on a branch — these are the only ideas that change the partition model fundamentally and could push past the 3ms ceiling on SCF. 
 
Idea #13 is the single biggest theoretical win but is the same shape of risk as the rejected XC-grid schedule — needs the float32+DIIS interaction studied first.