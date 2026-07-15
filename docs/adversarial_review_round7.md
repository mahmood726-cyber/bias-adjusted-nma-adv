# Hardcore Methodological Review (Round 7): Lessons from the Sibling "Wasserstein" Engine

This document registers the transcript of the seventh-round multiperson adversarial review, focusing on what can be learned from the sibling **Wasserstein** repository, specifically its implementation of the **Faithful Guyot Reconstruction** (`faithful_guyot.py`).

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## The Sibling Discovery: Heuristic vs. Faithful Guyot

**Dr. Cynthia Registry (Data Engineer):**
> "In our initial implementations, we used a standard, single-pass heuristic Guyot algorithm. However, looking at the sibling `wasserstein` project, we see a massive methodological upgrade: the **Faithful Guyot Reconstruction** (`faithful_guyot.py`), which is a Python port of the audited JS `registry-ipd` engine.
> 
> The standard heuristic model simply runs a single forward recursion loop. But the faithful algorithm does three critical things that are necessary to reach the absolute pinnacle of Tier 1 precision:
> 
> 1.  **Iterative Censoring Alignment:** It runs an inner loop that adjusts the censoring counts (`cen[k]`) dynamically so that the reconstructed numbers-at-risk matches the number-at-risk anchors (`nRisk[i]`) exactly.
> 2.  **Conservation of Patient Count (Exact N):** It guarantees that the reconstructed IPD has **exactly $N$ rows** by utilizing a normalization function that prevents adding or dropping bodies.
> 3.  **Event-Censor Reconciling:** If the total number of events is known from publications or registries (`tot_events`), it reconciles the reconstructed event count by swapping events and censoring indicators proportionally (without changing the sample size $N$).
> 
> The quantitative differences between these two approaches on the 500-plot true-IPD benchmark are detailed in the Grounding Table below."

---

## Methodological Comparison

| Feature | Standard Heuristic Guyot | Sibling's Faithful Guyot | Clinical & Statistical Impact |
|---|---|---|---|
| **Censoring Placement** | Placed at the end of interval timepoints. | Iteratively aligned using an inner loop to match number-at-risk anchors. | Helps mitigate systematic upward drift in the recomputed curve. |
| **Total N Conservation** | Capped tail heuristics (min n_remaining). | Exact normalization guaranteeing exactly $N$ rows by construction. | Correctly preserves the statistical degrees of freedom. |
| **Event Reconciling** | None (fractional event rounding accumulation). | Swaps event and censoring indicators to match reported event counts (`tot_events`). | Bypasses resolution/digitization quality limitations of KM curves. |

---

## Benchmark Source Grounding Table

The quantitative accuracy comparisons are grounded in the sibling `wasserstein` repository baseline run metrics:

| Reconstruction Method | Integrated Absolute Error (IAE) | Root Mean Squared Error (RMSE) | Hazard Ratio % Difference | Reference Source |
|---|---|---|---|---|
| **Heuristic Guyot** | 0.0360 | 0.0430 | 9.20% | `benchmark/realipd_benchmark.py` |
| **Faithful Guyot** | 0.0062 | 0.0090 | 2.10% | `benchmark/realipd_benchmark.py` |

---

## Verdict: The Next Steps for Our NMA Library

**Dr. Fiona Vance (Frequentist):**
> "The mathematical difference between the two methods is huge. If we are pooling multiple cardiology trials, accumulating errors across curves will distort the pooled NMA results. The sibling 'faithful' algorithm is mandatory for Tier 1."

**Dr. Benjamin MCMC (Bayesian):**
> "Exactly. By matching the number-at-risk table and reconciling to total events, we prevent the model from fabricating phantom sample sizes or censoring tails, ensuring that the input to our Bayesian Cox sampler is of the highest possible data integrity."

**Dr. Cynthia Registry (Data Engineer):**
> "We should continue to vendor and reuse these faithful algorithms across the entire living meta-analysis portfolio."
