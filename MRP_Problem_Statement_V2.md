# MRP Problem Statement (V2 — Post Deep Research)
## Task Arithmetic for Zero-Shot Indian Language Information Retrieval: A Systematic Study with Advanced Merging Methods

**Student:** Dharmik Mistry (202511039)  
**Program:** M.Tech ICT (Machine Learning), Semester 3  
**Supervisor:** Prof. Sandip Modha  
**Institute:** DA-IICT, Gandhinagar  
**Date:** July 2026  

---

## 1. Introduction & Motivation

Large Language Models fine-tuned for information retrieval (IR) achieve strong performance on English benchmarks but degrade significantly on unseen domains and languages. Adapting these models to new languages traditionally requires expensive supervised fine-tuning with domain-specific labeled data — a resource that is scarce or nonexistent for most Indian languages.

Task Arithmetic (Ilharco et al., ICLR 2023) offers a training-free alternative: extract a "task vector" by subtracting a base pretrained model's weights from a language-specialized model's weights, then add this vector to an IR-tuned model. The resulting model acquires language-specific knowledge without any additional training.

Braga et al. (SIGIR 2025) applied this framework to IR for the first time, demonstrating gains of up to 18% in NDCG@10 on European language retrieval tasks. However, their work has three significant limitations:

1. **Only European languages tested** — German, French, Spanish, and English. Indian languages present fundamentally harder transfer challenges: non-Latin scripts (Devanagari, Tamil, Bengali), severe subword fragmentation under multilingual tokenizers, SOV word order, rich agglutinative morphology, and pervasive code-mixing between English and native languages.

2. **Only vanilla weighted addition tested** — The paper uses the simplest merging strategy: Θ′ = Θ_T + α·τ_D with a single global scalar α. While recent work has begun exploring advanced merging methods (TIES-Merging, DARE, SLERP) for dense retrieval (Sasaki et al., CIKM 2025; Zhang et al., 2026), no study has systematically compared these conflict-aware methods against vanilla task arithmetic for cross-encoder re-ranking in a multilingual setting.

3. **No code-mixed or Romanized text** — Indian users frequently search in code-mixed language (e.g., "heart attack ke early signs batao"), a phenomenon that causes 10–15 point retrieval degradation even in strong multilingual models (Zeng et al., ACL 2026 Findings). Task arithmetic has never been applied to this challenge.

This project addresses all three gaps by providing the first systematic evaluation of task arithmetic — and its advanced variants — for Indian language information retrieval.

---

## 2. Base Paper

Marco Braga, Pranav Kasela, Alessandro Raganato, and Gabriella Pasi. 2025. "Investigating Task Arithmetic for Zero-Shot Information Retrieval." In Proceedings of SIGIR 2025, Padua, Italy. DOI: 10.1145/3726302.3730216.

**Core method:** Given a base pretrained model Θ₀, a domain/language-fine-tuned model Θ_D, and an IR-fine-tuned model Θ_T (all sharing the same architecture), the task vector τ_D = Θ_D − Θ₀ captures language-specific knowledge. Adding it to the IR model produces a language-aware retrieval model: Θ′ = Θ_T + α·τ_D.

**Pipeline:** BM25 first-stage retrieval (top 100 documents) → Cross-encoder re-ranking with Θ′.

**Key results:** Up to 18% NDCG@10 improvement on multilingual retrieval; statistically significant gains on TREC-COVID MAP@100; optimal α varies by model family (0.3–0.5 for smaller models, 0.7–0.9 for larger models).

---

## 3. Related Work

### 3.1 Model Merging Foundations

Task Arithmetic (Ilharco et al., ICLR 2023) introduced the concept of task vectors — parameter-space differences between fine-tuned and base models that can be added or subtracted to transfer capabilities. Several methods improve upon vanilla addition by resolving parameter conflicts: TIES-Merging (Yadav et al., NeurIPS 2023) prunes low-magnitude parameters and resolves sign disagreements before merging; DARE (Yu et al., ICML 2024) randomly drops task vector elements and rescales to preserve expectations; SLERP performs spherical linear interpolation in weight space; and AdaMerging (Yang et al., ICLR 2024) learns layer-wise and task-wise scaling factors adaptively.

### 3.2 Model Merging for IR

The application of model merging to information retrieval is recent and limited. Sasaki et al. (CIKM 2025, arXiv 2509.21966) apply merging techniques including SLERP, TIES, DARE, and Task Arithmetic to e5-mistral-7b-instruct for domain-specific ad-hoc retrieval. Zhang et al. (arXiv 2602.04731) use MergeKit with linear and TIES merging on BEIR dense retrievers. Han et al. (arXiv 2507.06782) merge time-stamped retrievers for temporal IR. However, no study has compared these methods for multilingual cross-encoder re-ranking, and none has tested on Indian languages.

The base paper authors (Braga et al.) have published a follow-up, DRAMA (Kasela et al., arXiv 2602.14960), on domain retrieval via adaptive module allocation — a different approach from task arithmetic that uses gated modular selection rather than weight-space merging.

### 3.3 Cross-Lingual Model Merging

Parović et al. (EACL 2024) investigate task arithmetic for cross-lingual transfer in NLU tasks. Chronopoulou et al. (MRL@EMNLP 2024) combine language and task arithmetic with parameter-efficient layers for zero-shot summarization. Klimaszewski et al. (COLING 2025, arXiv 2404.15737) propose language arithmetic for training-free adapter enhancement. AdaMergeX (Zhao et al., NAACL 2025) performs cross-lingual adapter merging including Hindi. Rafkin et al. (arXiv 2601.07038) apply task arithmetic with support languages for low-resource ASR. None of these works address information retrieval.

### 3.4 Indian Language IR

Indian language IR benchmarks are emerging but underserved. Hindi-BEIR (Acharya et al., 2024, arXiv 2409.05401) provides 15 retrieval datasets across 7 tasks. INDIC-MARCO (Haq et al., ACL 2024, arXiv 2312.09508) offers machine-translated MS-MARCO passages in 11 Indian languages. MIRACL (Zhang et al., 2023) includes human-annotated Hindi and Bengali subsets. Despite these resources, no published work applies task arithmetic or model merging to Indian language retrieval.

### 3.5 Code-Mixed Retrieval

Code-switching causes significant retrieval degradation across all model families (Zeng et al., ACL 2026 Findings, arXiv 2604.17632). The CMIR shared task at FIRE 2024–2025 addresses code-mixed Bengali-English retrieval, with best systems using XLM-R/MuRIL with transliteration. Kodali et al. (CODS-COMAD 2025, arXiv 2510.19782) apply task arithmetic and TIES-Merging to code-mixed Hindi-English classification, but not to retrieval. No work combines model merging with code-mixed IR.

---

## 4. Proposed Research

### 4.1 Problem Statement

> We investigate task arithmetic for zero-shot Indian language information retrieval, providing the first evaluation of language-specific task vectors for cross-encoder re-ranking in Hindi, Marathi, and code-mixed Hinglish. We further conduct the first systematic comparison of conflict-aware merging methods (TIES-Merging, DARE, SLERP) against vanilla task arithmetic for multilingual re-ranking, testing whether these methods — which resolve parameter conflicts during merging — improve upon simple addition when the source and target languages are typologically distant and use non-Latin scripts.

### 4.2 Research Questions

**RQ1:** Does task arithmetic effectively transfer Indian language knowledge into a multilingual cross-encoder re-ranker for zero-shot retrieval, and how does performance vary across languages with different scripts and morphological properties?

**RQ2:** Do conflict-aware merging methods (TIES-Merging, DARE, SLERP) outperform vanilla task arithmetic for Indian language cross-encoder re-ranking, and which method best handles the subword fragmentation and script diversity characteristic of Indic settings?

### 4.3 Stretch Goal (if time permits)

**RQ3:** Can task vectors from code-mixed pretrained models (HingRoBERTa, trained on 52.93M Hinglish sentences) improve retrieval for Romanized / code-mixed queries without any code-mixed IR training data?

---

## 5. Methodology

### 5.1 Three-Step Procedure (Following Braga et al.)

1. **Task Vector Generation:** Compute τ_D = Θ_D − Θ₀ by subtracting the base model's weights from the language-fine-tuned model's weights.

2. **Task Vector Integration:** Merge τ_D with the IR model Θ_T using one of the following methods:
   - **Vanilla Task Arithmetic:** Θ′ = Θ_T + α·τ_D (base paper's method)
   - **TIES-Merging:** Prune low-magnitude parameters → resolve sign conflicts → merge (Yadav et al., NeurIPS 2023)
   - **DARE:** Random dropout of task vector elements → rescale to preserve expectation → merge (Yu et al., ICML 2024)
   - **DARE-TIES:** Combine DARE sparsification with TIES sign resolution
   - **SLERP:** Spherical linear interpolation between Θ_T and (Θ₀ + τ_D)

3. **Zero-Shot Evaluation:** Evaluate Θ′ on Indian language retrieval benchmarks using BM25 first-stage retrieval → cross-encoder re-ranking.

### 5.2 Scaling Factor Protocol

Following the base paper, α is tuned via grid search from 0.1 to 1.0 in steps of 0.1. For datasets with development sets (MIRACL Hindi, Hindi-BEIR subsets with train splits), α is optimized on the dev set. For datasets without development sets, we report results at α = 1 (fully zero-shot). For TIES-Merging, the density parameter ρ is additionally swept over {0.1, 0.3, 0.5, 0.7, 0.9}. For DARE, the dropout rate p is swept similarly.

### 5.3 Evaluation Metrics

P@10, NDCG@3, NDCG@10, MAP@100 — matching the base paper. Statistical significance assessed via Bonferroni-corrected two-sided paired Student's t-test at 99% confidence.

---

## 6. Experimental Setup

### 6.1 Model Triplets

**Primary Family: XLM-RoBERTa-base (Encoder-Only)**

All models share FacebookAI/xlm-roberta-base as the backbone (hidden_size=768, 12 layers, 12 attention heads, vocab_size=250002, SentencePiece tokenizer).

| Role | Model | HuggingFace ID | Language |
|------|-------|---------------|----------|
| Θ₀ (Base) | XLM-RoBERTa-base | FacebookAI/xlm-roberta-base | 100 languages |
| Θ_T (IR) | XLM-R Cross-Encoder | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | mMARCO French (multilingual capability) |
| Θ_D (Hindi) | HindRoBERTa | l3cube-pune/hindi-roberta | Hindi |
| Θ_D (Hinglish) | HingRoBERTa | l3cube-pune/hing-roberta | Code-mixed Hindi-English |
| Θ_D (Mixed-script) | HingRoBERTa-Mixed | l3cube-pune/hing-roberta-mixed | Mixed-script Hinglish |
| Θ_D (Marathi) | MahaRoBERTa | l3cube-pune/marathi-roberta | Marathi |

**Secondary Family: MT5-base (Encoder-Decoder — For Baseline Reproduction)**

| Role | Model | HuggingFace ID | Language |
|------|-------|---------------|----------|
| Θ₀ (Base) | MT5-base | google/mt5-base | 101 languages |
| Θ_T (IR) | MT5-mMARCO | unicamp-dl/mt5-base-mmarco-v2 | mMARCO multilingual |
| Θ_D (German) | mT5-base-german | Calizzano et al. (2022) | German |
| Θ_D (French) | mT5-base-french | Calizzano et al. (2022) | French |
| Θ_D (Spanish) | mT5-base-spanish | Calizzano et al. (2022) | Spanish |

The MT5 family is used to reproduce the base paper's European language results as a baseline for comparison. No Indian-language mT5-base continued-pretraining checkpoints currently exist on HuggingFace.

### 6.2 Datasets

**For Indian Language Evaluation (Primary):**

| Dataset | Language | Qrels Type | Source |
|---------|----------|-----------|--------|
| MIRACL Hindi (dev) | Hindi | Human-annotated | Zhang et al., 2023 |
| Hindi-BEIR | Hindi (15 datasets) | Mixed (human + translated) | Acharya et al., 2024 |
| INDIC-MARCO Hindi | Hindi | Machine-translated | Haq et al., 2024 |
| INDIC-MARCO Marathi | Marathi | Machine-translated | Haq et al., 2024 |
| MIRACL Bengali (dev) | Bengali | Human-annotated | Zhang et al., 2023 |

**For Baseline Reproduction:**

| Dataset | Domain/Language | Source |
|---------|----------------|--------|
| TREC-COVID | Biomedical (English) | BEIR benchmark |
| NFCorpus | Medical (English) | BEIR benchmark |
| SCIDOCS | Scientific (English) | BEIR benchmark |
| SciFact | Fact-checking (English) | BEIR benchmark |
| GermanQuAD | German | Base paper |
| MIRACL French | French | Base paper |
| MIRACL Spanish | Spanish | Base paper |

### 6.3 Baselines

For each dataset, we compare against:
- **BM25** — lexical baseline
- **Θ₀ (Pre-trained)** — base model without any fine-tuning
- **Θ_D (Language-specific)** — language model without IR fine-tuning
- **Θ_T (IR-specific)** — MS-MARCO fine-tuned model without language adaptation
- **Θ′ Vanilla TA (α = 1)** — fully zero-shot task arithmetic
- **Θ′ Vanilla TA (optimized α)** — task arithmetic with tuned α
- **Θ′ TIES** — TIES-Merging with tuned density and α
- **Θ′ DARE** — DARE with tuned dropout rate and α
- **Θ′ SLERP** — Spherical interpolation with tuned interpolation coefficient

### 6.4 Implementation

- **Model merging:** MergeKit (arcee-ai/mergekit) for TIES, DARE, SLERP; custom PyTorch script for vanilla task arithmetic
- **BM25 retrieval:** Pyserini
- **Cross-encoder scoring:** HuggingFace Transformers (AutoModelForSequenceClassification)
- **Evaluation:** pytrec_eval / ir_measures for P@10, NDCG@3/10, MAP@100
- **Compute:** NVIDIA H200 and RTX 6000 (confirmed access at DA-IICT)

---

## 7. Expected Contributions

1. **First application of task arithmetic to Indian language information retrieval** — demonstrating whether language-specific task vectors from Indic pretrained models improve cross-encoder re-ranking for Hindi, Marathi, and code-mixed Hinglish

2. **First systematic comparison of conflict-aware merging methods for multilingual re-ranking** — evaluating TIES-Merging, DARE, and SLERP against vanilla task arithmetic on typologically diverse languages with non-Latin scripts, extending beyond the dense retrieval setting studied by Sasaki et al. (CIKM 2025)

3. **Analysis of failure conditions** — investigating when and why task arithmetic fails for Indian languages, including the effects of subword fragmentation, script mismatch, and morphological divergence

4. **Practical guidelines** — recommendations on which merging method works best under which conditions (language family, script type, model architecture)

---

## 8. Experimental Plan & Timeline

| Phase | Tasks | Duration | Deliverable |
|-------|-------|----------|-------------|
| **Phase 1: Setup** | Clone base paper code, set up evaluation pipeline, download datasets, verify model triplet compatibility | Weeks 1–2 | Working BM25 + cross-encoder pipeline on one BEIR dataset |
| **Phase 2: Reproduction** | Reproduce base paper's European results with MT5-base on GermanQuAD, MIRACL French/Spanish | Weeks 3–4 | Reproduction table matching Table 2 of base paper |
| **Phase 3: Indian Language TA** | Run vanilla task arithmetic with XLM-R family on Hindi-BEIR, MIRACL Hindi, INDIC-MARCO Hindi/Marathi | Weeks 5–7 | Results table for RQ1 |
| **Phase 4: Advanced Merging** | Implement TIES, DARE, SLERP via MergeKit; run on all datasets from Phase 3 | Weeks 8–10 | Comparison table for RQ2 |
| **Phase 5: Code-Mixed (stretch)** | Run task arithmetic with HingRoBERTa on code-mixed evaluation | Weeks 11–12 | Preliminary results for RQ3 |
| **Phase 6: Analysis & Writing** | Statistical significance testing, failure analysis, paper writing | Weeks 13–16 | Draft paper + MRP thesis |

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| XLM-R cross-encoder (French-trained) degrades on Hindi re-ranking | Medium | High | Test Θ_T alone on Hindi first; if it fails, we document the failure condition — itself a finding |
| TIES/DARE show no improvement over vanilla TA | Medium | Low | Negative result is publishable: "conflict-aware merging is unnecessary for cross-lingual re-ranking" is a clean finding |
| L3Cube models don't produce valid task vectors (architecture mismatch in config) | Low | Critical | Verify config.json match before starting experiments; fallback to mBERT family |
| Hindi-BEIR translation artifacts confound results | Medium | Medium | Prioritize MIRACL Hindi (human-annotated qrels) as primary evaluation; report Hindi-BEIR as secondary |
| Compute bottleneck on large-scale BEIR evaluation | Low | Medium | H200 and RTX 6000 confirmed; encoder models (~278M params) are manageable |
| Code-mixed evaluation data too thin for publishable results (RQ3) | High | Low | RQ3 is a stretch goal, not core; project succeeds without it |

---

## 10. Fallback Plan

If advanced merging methods (RQ2) show no gains over vanilla task arithmetic, the project degrades gracefully to RQ1 alone — a rigorous evaluation of vanilla task arithmetic for Indian language IR. This remains novel (no prior work exists) and publishable. The finding "vanilla α-scaling is as good as TIES/DARE for cross-lingual re-ranking" is a clean, defensible result that saves the community from unnecessary complexity.

---

## 11. Key References

1. Braga et al. (SIGIR 2025) — Base paper. DOI: 10.1145/3726302.3730216
2. Ilharco et al. (ICLR 2023) — Original task arithmetic. arXiv: 2212.04089
3. Yadav et al. (NeurIPS 2023) — TIES-Merging. arXiv: 2306.01708
4. Yu et al. (ICML 2024) — DARE. arXiv: 2311.03099
5. Sasaki et al. (CIKM 2025) — Model merging for domain-specific retrieval. arXiv: 2509.21966
6. Zhang et al. (2026) — Less Finetuning, Better Retrieval. arXiv: 2602.04731
7. Kasela et al. (2026) — DRAMA (base paper authors' follow-up). arXiv: 2602.14960
8. Parović et al. (EACL 2024) — Task arithmetic for cross-lingual transfer
9. Chronopoulou et al. (MRL@EMNLP 2024) — Language and task arithmetic for summarization
10. Kodali et al. (CODS-COMAD 2025) — Task arithmetic for code-mixed classification. arXiv: 2510.19782
11. Zeng et al. (ACL 2026 Findings) — Code-switching IR benchmarks. arXiv: 2604.17632
12. Acharya et al. (2024) — Hindi-BEIR. arXiv: 2409.05401
13. Haq et al. (ACL 2024) — IndicIRSuite / INDIC-MARCO. arXiv: 2312.09508
14. Nayak & Joshi (WILDRE 2022) — L3Cube-HingCorpus and HingBERT. arXiv: 2204.08398
15. Zhang et al. (TACL 2023) — MIRACL multilingual retrieval dataset
16. Calizzano et al. (LREC 2022) — mT5 language-specific checkpoints
17. Yang et al. (ICLR 2024) — AdaMerging. arXiv: 2310.02575

---

*V2 — Corrected after deep research validation on GPT Deep Research, Gemini Deep Research, and Claude. All novelty claims verified against existing literature. False "zero papers" claim corrected with proper citations (Sasaki et al., Zhang et al.). Direction D (multi-vector composition) downgraded due to prior art (Parović, Chronopoulou, Klimaszewski). XLM-RoBERTa identified as primary model family after MT5 Indian language checkpoint gap discovered.*
