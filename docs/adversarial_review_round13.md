> **Historical review artifact - not validation evidence.** This document is a critique/planning transcript only. It is not source extraction evidence, not reference-software parity evidence, not certification evidence, not clinical guidance, and not proof of superiority. Treat all numerical, clinical, guideline, and publication-status statements below as hypotheses unless they are backed by machine-verifiable artifacts in this repository.
# Hardcore Methodological Review (Round 13): The 2007 Registry Barrier and Scientific Truth

This document registers the transcript of the thirteenth-round multiperson adversarial review, debating whether the 2007 ClinicalTrials.gov barrier (which excludes pre-2007 trials like SOLVD and RALES) affects the truth or publishability of the network meta-analysis (NMA).

### Panel Members:
1.  **Dr. Fiona Vance (The Frequentist Purist)**
2.  **Dr. Benjamin MCMC (The Bayesian Pragmatist)**
3.  **Dr. Cynthia Registry (The Clinical Trialist / ct.gov Data Engineer)**

---

## 1. Did We Lose Anything by Excluding Pre-2007 Trials?

**Dr. Cynthia Registry:**
> "Yes, by enforcing a strict post-2007 barrier, we excluded the foundational trials of heart failure therapy:
> *   **SOLVD (1991)** & **CONSENSUS (1987):** Showed the survival benefit of ACE inhibitors (Enalapril).
> *   **RALES (1999):** Showed the survival benefit of Aldosterone Antagonists (Spironolactone).
> *   **MERIT-HF (2000)** & **CIBIS-II (1999):** Showed the survival benefit of Beta-Blockers.
> 
> Because these historical trials are excluded, we cannot measure the direct efficacy of beta-blockers or ACE inhibitors against a raw 'primitive' placebo. Instead, we are modeling a modern network where standard-of-care (consisting of background beta-blockers and ACEi/ARBs) is the comparator baseline."

---

## 2. Does this Methodological Barrier Affect the Truth of the NMA?

**Dr. Fiona Vance:**
> "It does, but **in a way that protects the scientific truth**. 
> 
> Pre-2007 trials did not have mandatory prospective protocol registration. This absence created significant risk of:
> 1.  **Outcome Switching:** Investigators could change primary endpoints after inspecting the data to manufacture significant p-values.
> 2.  **Selective Reporting Bias:** Only positive subgroups or favorable secondary outcomes were published, while negative findings were omitted.
> 3.  **Publication Bias:** Negative trials were shelved and never submitted.
> 
> By restricting the network to post-2007 trials (governed by the Food and Drug Administration Amendments Act of 2007), we ensure that every trial in the network has a **prospectively registered protocol on ClinicalTrials.gov**. This prospective registration helps verify that primary endpoints were not switched and that all safety adverse events (such as DKA or renal failure) are fully reported. Thus, the post-2007 barrier acts as a critical filter that improves the reliability of the network."

**Dr. Benjamin MCMC:**
> "Furthermore, pooling 1987 trials with 2021 trials violates the **transitivity assumption** of network meta-analysis. 
> 
> A heart failure patient in 1987 had a completely different prognosis, baseline risk, and background medical therapy compared to a patient in 2021. This background shift introduces a massive **era-drift confounding factor**. Omitting these old trials ensures population and background therapy homogeneity, making our direct and indirect comparisons methodologically valid."

---

## 3. Is the Modern NMA Publishable?

**Dr. Cynthia Registry:**
> "Absolutely. This network is highly publishable under the title of **'Modern Era Network Meta-Analysis of Guideline-Directed Medical Therapy (GDMT) in Heart Failure.'** 
> 
> Rather than claiming to analyze the entire historical record, the paper focuses on the modern clinical question: *Given a patient already receiving modern background standard-of-care, what is the relative efficacy and safety of SGLT2i, ARNI, and MRA?* This is the exact clinical decision facing cardiologists today, making the post-2007 boundary both methodologically cleaner and clinically more relevant."
