# MRP Research Report: Task Arithmetic for Zero-Shot Information Retrieval
## Complete Project Documentation — From Initiation to Current Status

**Student:** Dharmik Mistry (202511039)  
**Program:** M.Tech ICT (Machine Learning), DA-IICT, Gandhinagar  
**Supervisor:** Prof. Sandip Modha (IR, LLM, NLP)  
**Date of Report:** July 8, 2026  
**Report Version:** 1.0  
**Status:** Pre-definition submission — problem statement under development  

---

# PART I: PROJECT INITIATION & BASE PAPER ANALYSIS

## 1.1 Project Origin

Prof. Sandip Modha assigned the SIGIR 2025 paper "Investigating Task Arithmetic for Zero-Shot Information Retrieval" by Braga et al. as the base paper for the Major Research Project (MRP). The professor's specific instruction was: **"Take this base paper and find something novel we can do with Indian languages."**

The MRP spans Semester 3 (approximately July–November 2026). The project definition document is due for submission imminently. The full project is due by November 2026. A publication is expected but the venue has not been specified — Prof. Modha will decide. Available compute resources include confirmed access to NVIDIA H200 and RTX 6000 GPUs at DA-IICT.

A parallel project (LLM personalization on LaMP benchmark — CS-SV cold-start activation steering) is running simultaneously, which constrains available research hours to approximately 12–18 focused hours per week for this MRP.

## 1.2 Base Paper: Complete Analysis

**Full Citation:**  
Marco Braga, Pranav Kasela, Alessandro Raganato, and Gabriella Pasi. 2025. "Investigating Task Arithmetic for Zero-Shot Information Retrieval." In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13–18, 2025, Padua, Italy. ACM, New York, NY, USA, 6 pages. DOI: 10.1145/3726302.3730216. arXiv: 2505.00649.

**Code Repository:** https://github.com/DetectiveMB/Task-Arithmetic-for-ZS-IR (verified accessible — contains Python scripts, utilities, and references to BEIR/MIRACL data usage; no model checkpoints, just training/evaluation scripts).

### 1.2.1 Core Idea

Instead of fine-tuning an IR model on every new domain or language (expensive, requires labeled data), the paper extracts a "Task Vector" — the parameter difference between a domain-fine-tuned LLM and its base pretrained version — and adds it to an IR-fine-tuned model. This produces a domain-aware IR model with zero additional training.

### 1.2.2 The Three-Model Setup

Task Arithmetic requires three models sharing the exact same base architecture:

- **Θ₀** — Base pretrained LLM (e.g., T5-base, LLaMA-2-7B, MT5-base)
- **Θ_D** — Θ₀ further fine-tuned on a specific domain or language (e.g., SciFive for biomedical, MedTuned for clinical, mT5-base-german for German)
- **Θ_T** — Θ₀ fine-tuned on MS-MARCO for IR (e.g., MonoT5, RankingGPT-LLaMA, msmarco-RoBERTa)

### 1.2.3 Mathematical Formulation

**Task Vector Generation (Equation 1):**  
τ_D = {τ₁, ..., τ_N}, where τᵢ = (θᵢ)_D − (θᵢ)₀

This captures the domain-specific shift in parameter space — what the domain fine-tuning "learned."

**Task Vector Integration (Equation 2):**  
Θ′ = {θ′ᵢ = (θᵢ)_T + α·τᵢ} for i = 1 to N

The scaling factor α ∈ ℝ controls how much domain knowledge is injected. At α = 0, Θ′ defaults to Θ_T. At α > 0, specialized knowledge is added. At α < 0, it is subtracted.

### 1.2.4 Pipeline

1. **BM25 first-stage retrieval** — retrieve top 100 documents
2. **Cross-encoder re-ranking** — re-rank using the merged model Θ′
3. **Final score** — weighted sum of BM25 and LLM scores: λ_BM25 and λ_LLM optimized in [0,1] on development sets; default λ_BM25 = λ_LLM = 0.5 when no dev set is available

### 1.2.5 Experimental Setup

**Models tested (6 total):**
- Encoder-only bi-encoders: DistilBERT, RoBERTa-base
- Encoder-decoder cross-encoders: T5-base, T5-Large, MT5-base
- Decoder-only LLM: LLaMA-2-7B

**Domain models (Θ_D):**
- Biomedical/Scientific: LLaMA2-MedTuned-7b, SciFive, BioMed-RoBERTa, Bio-DistilBERT
- Language-specific: mT5-base-german, mT5-base-spanish, mT5-base-french, mT5-base-english (Calizzano et al., 2022)

**IR models (Θ_T):**
- RankingGPT-LLaMA2-7b, MonoT5, msmarco-RoBERTa, msmarco-distilbert, MT5-base-msmarco

**Datasets (8 total):**
- BEIR Biomedical/Scientific: TREC-COVID, NFCorpus, SCIDOCS, SciFact
- Multilingual: GermanQuAD, MIRACL Spanish, MIRACL French, MIRACL English

**Metrics:** P@10, NDCG@3, NDCG@10, MAP@100  
**Significance testing:** Bonferroni-corrected two-sided paired Student's t-test at 99% confidence  
**α tuning:** Grid search from 0.1 to 1.0 in steps of 0.1; tuned on NFCorpus dev + 20% SciFact training queries; fully zero-shot (α = 1) when no dev set is available

### 1.2.6 Key Results

**Biomedical/Scientific (Table 1):**
- At α = 1 (fully zero-shot): Task Arithmetic outperforms MS-MARCO baselines only on TREC-COVID with RoBERTa-base, T5-base, and T5-Large, and on SCIDOCS with T5-Large
- With optimized α: Consistent improvements across all models and datasets, with statistically significant gains in MAP@100 on TREC-COVID and NDCG@10 on TREC-COVID and SCIDOCS
- Optimal α exceeds 0.3 for all models, surpasses 0.7 for T5 variants and LLaMA-2
- Smaller models (DistilBERT, RoBERTa) peak at α ∈ [0.3, 0.5]; larger models (T5, LLaMA) prefer α ∈ [0.7, 0.9]

**Multilingual (Table 2):**
- MT5-base with language-specific task vectors: Statistically significant improvements over the IR-specific model by up to **18% in NDCG@10** across all four language tasks
- Pre-trained MT5-base (Θ₀) and language-specific variant (Θ_D) do not outperform BM25 alone — confirming these models are not inherently optimized for IR

**Ablation Study (Table 3):**
- Performance varies considerably with α — e.g., T5-base on SciFact: .640 at α=1.0 vs .722 at α=0.7 (12% difference)
- α = 1.0 rarely provides best performance — excessive domain emphasis overshadows IR-specific knowledge
- No single α is optimal across all models or datasets

### 1.2.7 Explicit Limitations of the Base Paper

1. **Only vanilla addition tested** — no advanced merging methods (TIES, DARE, SLERP). Paper cites TIES-Merging (ref [63]) but never compares against it.
2. **Only European languages tested** for multilingual transfer — German, French, Spanish, English
3. **Only single task vector addition** — no multi-vector composition (domain + language simultaneously)
4. **Only cross-encoder re-ranking** — no first-stage dense retrieval
5. **Global α** — same α applied uniformly to all queries and all layers
6. **Only MS-MARCO as IR training source**
7. **No Indian languages, no code-mixed text, no non-Latin scripts**

---

# PART II: IDEA GENERATION & INITIAL FILTERING

## 2.1 Initial Ideas Generated (Pre-Deep-Research)

The following ideas were generated through analysis of the base paper's limitations and gaps:

### Idea 1: Query-Adaptive Scaling (α as a function of query)
**Concept:** Instead of a single global α, make α(q) dependent on the query. Use a signal like term overlap with domain-specific vocabulary or perplexity gap under the domain model vs. base model.  
**Status:** REJECTED  
**Reason:** AdaMerging (Yang et al., ICLR 2024) already does adaptive scaling via entropy minimization on test inputs, with both task-wise and layer-wise variants. Applying it to IR has some novelty but the core idea is not new — it's applying an existing technique to a new domain, which is incremental.

### Idea 2: LoRA-Native Task Arithmetic
**Concept:** Instead of full-weight task vectors, operate in the LoRA adapter space (rank 16 or 64). LoRA adapters are already conceptually task vectors (ΔW = BA). Composition properties might differ in this compressed subspace.  
**Status:** REJECTED  
**Reason:** This is an extremely active and crowded area. LoRAHub, LoRA Soups, Task-Aware LoRA Composition via Retrieval (Feb 2026), rank-wise clustering for LoRA merging — all exist. HuggingFace PEFT library already supports TIES, DARE, and weighted averaging for LoRA adapters natively. Doing this for IR alone doesn't provide enough novelty.

### Idea 3: Indian Language IR (Plain Extension)
**Concept:** Simply run the base paper's exact method on Indian language datasets.  
**Status:** REJECTED as standalone (too incremental)  
**Reason:** Hindi-BEIR benchmark exists (Acharya et al., 2024). IndicIRSuite with INDIC-MARCO covering 11 Indian languages exists (Dec 2023). Simply running task arithmetic on Indian languages without methodological innovation = incremental extension that reviewers will see through.  
**Note:** Retained as the CORE SPINE of the project (Direction B) when combined with methodological depth.

### Idea 4: Layer-Wise Task Arithmetic for IR
**Concept:** Apply task vectors selectively to attention layers, FFN layers, or only top-K layers, with per-layer α values.  
**Status:** REJECTED  
**Reason:** LATA (Chen et al., Findings EMNLP 2025, arXiv 2502.20186) already does per-layer weight assignment based on cosine similarity analysis. AdaRank (2025) does layer-wise pruning. Not novel as a standalone idea.

### Idea 5: Cross-Lingual Model Merging (Broad)
**Concept:** Apply model merging for cross-lingual transfer broadly.  
**Status:** REJECTED as standalone  
**Reason:** AdaMergeX (Zhao et al., NAACL 2025) does cross-lingual adapter merging with Hindi. "The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer" (Bandarkar & Peng, MRL 2025, arXiv 2505.18356) exists. "No Train but Gain: Language Arithmetic" (Klimaszewski et al., COLING 2025, arXiv 2404.15737) already does this for language adapters.

### Idea 6: Advanced Merging Methods (TIES, DARE, SLERP) for IR ⭐
**Concept:** Systematically compare TIES-Merging, DARE, DARE-TIES, and SLERP against vanilla task arithmetic for IR re-ranking.  
**Initial Status:** Claimed as "CONFIRMED NOVEL — zero papers found"  
**Post-Deep-Research Status:** PARTIALLY NOVEL — reframed (see Section 3.1.1)  
**Final Status:** APPROVED as Direction A (methodological extension)

### Idea 7: Code-Mixed / Hinglish Retrieval via Task Arithmetic ⭐
**Concept:** Use models pretrained on Romanized code-mixed text (HingRoBERTa trained on 52.93M Hinglish sentences) as domain models for task arithmetic, enabling zero-shot code-mixed retrieval.  
**Initial Status:** Claimed as "genuinely untouched intersection"  
**Post-Deep-Research Status:** NOVEL for IR (but feasibility concerns — see Section 3.1.3)  
**Final Status:** APPROVED as stretch goal / future work (Direction C)

### Idea 8: Multi-Vector Composition (τ_domain + τ_language) for IR
**Concept:** Test whether Θ′ = Θ_T + α₁·τ_domain + α₂·τ_language enables simultaneous domain and language transfer for IR.  
**Initial Status:** Claimed as "unexplored for IR"  
**Post-Deep-Research Status:** PRE-EMPTED (see Section 3.1.4)  
**Final Status:** REJECTED as primary contribution (Direction D downgraded to future work)

### Idea 9: Task Arithmetic for Dense First-Stage Retrieval
**Concept:** Apply task vectors to bi-encoder dense retrieval models rather than cross-encoder re-rankers.  
**Status:** DEFERRED  
**Reason:** Changes the base paper's paradigm significantly. More engineering (indexing, ANN search). Good future work but out of scope for MRP.

## 2.2 Ideas Approved for Further Investigation (Pre-Deep-Research)

After initial filtering, four directions survived:

- **Direction A:** Advanced merging methods (TIES, DARE, SLERP) for IR
- **Direction B:** Indian language transfer via task arithmetic
- **Direction C:** Code-mixed / Hinglish retrieval via task arithmetic
- **Direction D:** Multi-vector composition (τ_domain + τ_language) for IR

These four were then subjected to deep research validation.

---

# PART III: DEEP RESEARCH PHASE — RECONCILED FINDINGS

## 3.0 Deep Research Methodology

Three separate deep research sessions were conducted simultaneously on July 8, 2026:

1. **GPT Deep Research** — Role: Novelty scout & resource discovery agent (autonomous multi-step searching, cross-referencing HuggingFace and academic papers)
2. **Gemini Deep Research** — Role: Literature landscape mapper & gap analyst (synthesis, structured reasoning, Google Search grounding)
3. **Claude (new conversation)** — Role: Strategic critic & problem statement generator (scope evaluation, honest assessment, reviewer simulation)

Tailored prompts were created for each platform, optimized for their respective strengths. The findings were then reconciled, with contradictions explicitly surfaced.

## 3.1 Novelty Validation — Reconciled Verdicts

### 3.1.1 Direction A: Advanced Merging Methods for IR

**Initial claim:** "Zero papers found applying TIES/DARE/SLERP to retrieval"

**GPT verdict:** PARTIALLY NOVEL — found Sasaki et al. (2025) applying weight-level model merging to domain-specific retrieval, and Han et al. (2025) proposing Time-Specifier Merging for temporal IR. Both use linear combination. Did NOT find TIES/DARE/SLERP specifically in IR.

**Gemini verdict:** NO papers found (gap confirmed) — stated that "no peer-reviewed literature has systematically applied conflict-resolving merging methods to dense retrieval, passage retrieval, or cross-encoder re-ranking pipelines."

**Claude verdict:** THE CLAIM IS FALSE. Found three papers:
1. **Sasaki et al., "Effect of Model Merging in Domain-Specific Ad-hoc Retrieval"** (arXiv 2509.21966, published at CIKM 2025, DOI 10.1145/3746252.3760920) — explicitly states merging "techniques, such as SLERP, TIES, DARE, and Task Arithmetic have also been proposed" and applies them to e5-mistral-7b-instruct retrievers
2. **"Less Finetuning, Better Retrieval"** (arXiv 2602.04731) — uses MergeKit with linear + TIES merging (sweeping density ρ ∈ {0.1…0.9}) on BEIR dense retrievers
3. **"Temporal Information Retrieval via Time-Specifier Model Merging"** (arXiv 2507.06782) — merges retrieval models

**RECONCILED VERDICT: The "zero papers" claim is FALSE and must be corrected before any submission.** Advanced merging for IR exists, but it has NOT been systematically compared for multilingual/Indian-language cross-encoder re-ranking. The defensible framing: "We provide the first controlled comparison of conflict-aware merging methods (TIES, DARE, SLERP) against vanilla task arithmetic for cross-lingual re-ranking in typologically distant, low-resource Indian languages."

**Key contradiction:** Gemini missed these papers entirely. GPT partially caught them. Claude was most thorough. This demonstrates why multi-platform validation was essential.

### 3.1.2 Direction B: Indian Language Task Arithmetic for IR

**GPT verdict:** NOVEL — no papers found applying task arithmetic/model merging to Indian-language IR.

**Gemini verdict:** NO papers found — confirmed the gap. Noted Kodali et al. (CODS-COMAD 2025, arXiv 2510.19782) applied task arithmetic and TIES-Merging to XLM-R and Llama-3.2-1B for English-Hindi code-mixed sentence CLASSIFICATION (sentiment, hate speech), but NOT IR.

**Claude verdict:** NOVEL — confirmed no prior work exists.

**RECONCILED VERDICT: CONFIRMED NOVEL.** No paper applies task arithmetic or model merging to Indian-language information retrieval. The closest work (Kodali et al., CODS 2025) applies TA + TIES to code-mixed classification, not retrieval. This is a genuine, defensible gap.

**Important related finding from Gemini:** Gemini identified three specific failure conditions for task arithmetic on Indian languages that must be addressed in the experimental design:
1. **Severe tokenization mismatch** — non-Latin scripts (Devanagari, Tamil) vs. Latin-centric base models
2. **Subword fragmentation** — multilingual tokenizers split non-Latin words into highly fragmented rare subwords, degrading vector alignment
3. **Morphological and syntactic divergence** — Indian languages use SOV word order (vs. SVO in European languages), and have rich morphological structure

### 3.1.3 Direction C: Code-Mixed Retrieval via Task Arithmetic

**GPT verdict:** NOVEL for IR — found Wang et al. (2025, arXiv) "Adapting Multilingual Models to Code-Mixed Tasks via Model Merging" but it tackles classification, not IR. No merging-based code-mixed IR found.

**Gemini verdict:** NOVEL for IR — found same Kodali et al. (CODS 2025) on classification. Code-mixed IR relies on hybrid pipelines (BM25 + multilingual transformers) and specialized encoders. CMIR at FIRE 2024-2025 uses XLM-R/MuRIL with transliteration or GPT-based reranking. No merging approaches.

**Claude verdict:** NOVEL for IR but HIGH RISK due to:
- Model triplet incompatibility (initially: no xlm-roberta-base MS-MARCO cross-encoder found — **LATER CORRECTED**, see Section 4.2)
- Evaluation data is stale and thin: FIRE MSIR (2014-2016, ~25 queries, decade old); CMIR 2025 (50 queries, Bengali-English, not Hinglish). No modern Hinglish retrieval benchmark with dense qrels exists.

**RECONCILED VERDICT: CONFIRMED NOVEL for IR, but evaluation-data-constrained.** The concept is untouched, but publishable evaluation is limited by the absence of a modern Hinglish IR benchmark. Suitable as a stretch goal, not as the core contribution. Model triplet feasibility was later resolved (see Section 4).

### 3.1.4 Direction D: Multi-Vector Composition for IR

**GPT verdict:** NOVEL — no literature found on adding multiple task vectors simultaneously for IR.

**Gemini verdict:** NOVEL for IR — ranked as "most novel" of the four directions.

**Claude verdict:** LARGELY PRE-EMPTED. Identified that the concept of τ_domain + τ_language composition is exactly the contribution of:
- Chronopoulou et al. (MRL@EMNLP 2024), "Language and Task Arithmetic with Parameter-Efficient Layers for Zero-Shot Summarization"
- Parović et al. (EACL 2024, Vol. 2 Short Papers, pp. 124-137), "Investigating the Potential of Task Arithmetic for Cross-Lingual Transfer" — which is CITED IN THE BASE PAPER ITSELF as ref [43]
- Klimaszewski et al. (COLING 2025, arXiv 2404.15737), "No Train but Gain: Language Arithmetic for training-free Language Adapters enhancement"

D is NOT novel as a concept; it would only be novel as "τ_domain + τ_language composition specifically for IR re-ranking," which is narrow.

**RECONCILED VERDICT: PRE-EMPTED. Downgraded to future work.** Gemini's "most novel" ranking was incorrect. Claude correctly identified the prior art. The key contradiction: Gemini ranked D #1 in novelty while Claude showed it's already done in NLP. This is why critical evaluation matters.

## 3.2 Scooping Risk Assessment

### 3.2.1 Base Paper Authors' Activity

**GPT finding:** No direct follow-ups by Braga/Kasela/Pasi on model merging for IR. Recent work includes adapter methods (MAdaKron adapters, KBC 2025) and domain-adaptive retrieval.

**Gemini finding (IMPORTANT):** Found **DRAMA: Domain Retrieval using Adaptive Module Allocation** by Pranav Kasela, Marco Braga, Ophir Frieder, Nazli Goharian, Gabriella Pasi, and Raffaele Perego (arXiv 2602.14960, 2026). This is a direct follow-up from the same group on adaptive domain retrieval, though it uses module allocation rather than task arithmetic.

**Reconciled assessment:** The Braga group IS actively working on domain-adaptive retrieval. DRAMA is not a direct scoop on our task-arithmetic approach, but indicates competition in the broader space. We should cite it and differentiate our work.

### 3.2.2 Indian NLP Groups

All three platforms found **no evidence** of Indian NLP groups (IITs, AI4Bharat, IISc, IIIT-H, L3Cube) working on model merging for IR. L3Cube focuses on creating pretrained models; AI4Bharat focuses on IndicTrans and IndicBERT; Indian IR groups at FIRE focus on traditional retrieval methods.

**One relevant finding:** Kodali et al. (CODS-COMAD 2025) at IIIT-Hyderabad applied task arithmetic + TIES to code-mixed classification. They could extend to IR. Low but non-zero scooping risk.

**Overall scooping risk: LOW.** The intersection of task arithmetic + Indian language + IR appears unclaimed.

## 3.3 Reviewer Objections (from Gemini's Simulation)

**Against Direction A (Advanced Merging):** "The student assumes that conflict-resolution methods optimized for cross-entropy minimization on multi-class classification tasks translate directly to pairwise rank-margin preservation. The paper lacks a theoretical analysis showing how pruning updates (DARE) or sign consensus (TIES) affects the semantic margin between relevant and irrelevant query-passage embedding pairs."

**Against Direction B (Indian Language Transfer):** "The lexical alignment of multilingual pre-trained models on non-Latin scripts is notoriously poor due to subword overfragmentation. The reported performance gains from task arithmetic on Devanagari or Tamil tasks are likely a byproduct of matching English-transliterated entities rather than true zero-shot cross-lingual semantic alignment. The evaluation must include a baseline with spelling-normalized or transliterated inputs to isolate these effects."

**Against Direction C (Code-Mixed Retrieval):** "The code-mixed benchmarks utilized are heavily influenced by colloquial spelling variations and non-standard orthography. Without incorporating an explicit spelling-normalization pre-processing step, the dense model is simply memorizing Romanized phonetic similarities. The evaluation fails to isolate the benefits of model merging from the benefits of standard vocabulary expansion on code-mixed data."

## 3.4 Minimum Viable Publishable Result

**Gemini's recommendation:** Demonstrate that TIES-Merging consistently and statistically outperforms vanilla task arithmetic from Braga et al. across BEIR benchmarks, with ≥1-2 absolute NDCG@10 improvement and p < 0.05.

**Claude's recommendation:** A method × language NDCG@10 matrix establishing whether any advanced merger beats vanilla TA on Indian languages, with significance tests. Even "vanilla α-scaling is as good as TIES/DARE for cross-lingual re-ranking" is a clean, defensible negative result.

**Reconciled:** The minimum viable result is a comprehensive comparison table (like Table 1 in the base paper) with rows for each merging method and columns for Indian language datasets. Negative results are publishable if rigorously analyzed.

---

# PART IV: MODEL TRIPLET DISCOVERY — COMPLETE INVENTORY

## 4.1 Critical Constraint

Task arithmetic requires that Θ₀, Θ_D, and Θ_T share the **exact same base architecture**: same hidden_size, num_hidden_layers, num_attention_heads, vocab_size, and tokenizer. If architectures don't match, task vector subtraction/addition is mathematically invalid — the parameter tensors have different shapes.

## 4.2 Family 1: XLM-RoBERTa-base (Encoder-Only)

**Architecture:** hidden_size=768, num_hidden_layers=12, num_attention_heads=12, vocab_size=250002

### Base Model (Θ₀)
| Model | HuggingFace ID | Parameters | Status |
|-------|---------------|------------|--------|
| XLM-RoBERTa-base | FacebookAI/xlm-roberta-base | ~278M | ✅ Verified |

### IR Models (Θ_T)
| Model | HuggingFace ID | Type | Training Data | Status |
|-------|---------------|------|--------------|--------|
| XLM-R Cross-Encoder FR | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | Cross-encoder (AutoModelForSequenceClassification) | mMARCO French | ✅ Verified — architecture matches (768 hidden, 12 layers, initialized from FacebookAI/xlm-roberta-base) |
| XLM-R Bi-Encoder MSMARCO | PaDaS-Lab/xlm-roberta-base-msmarco | Bi-encoder (SentenceTransformer) | MS-MARCO English | ✅ Verified — but bi-encoder paradigm, NOT cross-encoder re-ranking |

**Important note:** The cross-encoder (antoinelouis) was trained on French mMARCO. Since XLM-R supports 100 languages including Hindi, the model retains multilingual capability, but this introduces a French-language bias. This is a caveat, not a blocker — it could even be framed as part of the cross-lingual investigation. Discovery credit: this model was found by the student, correcting Claude's deep research claim that no such model exists.

### Indian Language Domain Models (Θ_D) — L3Cube Family
| Model | HuggingFace ID | Language | Base | Training Corpus | Status |
|-------|---------------|----------|------|----------------|--------|
| HindRoBERTa | l3cube-pune/hindi-roberta | Hindi | xlm-roberta-base | Hindi monolingual corpus | ✅ Verified |
| HingRoBERTa | l3cube-pune/hing-roberta | Hinglish (code-mixed) | xlm-roberta-base | L3Cube-HingCorpus (52.93M Hinglish sentences) | ✅ Verified |
| HingRoBERTa-Mixed | l3cube-pune/hing-roberta-mixed | Hinglish (mixed-script) | xlm-roberta-base | L3Cube-HingCorpus mixed script | ✅ Verified |
| MahaRoBERTa | l3cube-pune/marathi-roberta | Marathi | xlm-roberta-base | Marathi monolingual corpus | ✅ Verified |
| Gujarati RoBERTa | l3cube-pune/gujarati-roberta (?) | Gujarati | xlm-roberta-base (?) | — | ❓ NEEDS VERIFICATION |
| Bengali RoBERTa | ? | Bengali | xlm-roberta-base (?) | — | ❓ NEEDS VERIFICATION |
| Tamil RoBERTa | ? | Tamil | xlm-roberta-base (?) | — | ❓ NEEDS VERIFICATION |

### Confirmed Working Triplets (XLM-R Family)
| # | Θ₀ | Θ_D | Θ_T | Purpose |
|---|-----|------|------|---------|
| 1 | xlm-roberta-base | l3cube-pune/hindi-roberta (Hindi) | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | Hindi language transfer |
| 2 | xlm-roberta-base | l3cube-pune/hing-roberta (Hinglish) | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | Code-mixed retrieval |
| 3 | xlm-roberta-base | l3cube-pune/hing-roberta-mixed (Mixed-script Hinglish) | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | Mixed-script code-mixed |
| 4 | xlm-roberta-base | l3cube-pune/marathi-roberta (Marathi) | antoinelouis/crossencoder-xlm-roberta-base-mmarcoFR | Marathi language transfer |

## 4.3 Family 2: MT5-base (Encoder-Decoder — Base Paper's Family)

**Architecture:** Encoder-decoder, ~580M parameters

### Base Model (Θ₀)
| Model | HuggingFace ID | Parameters | Status |
|-------|---------------|------------|--------|
| MT5-base | google/mt5-base | ~580M | ✅ Verified |

### IR Models (Θ_T)
| Model | HuggingFace ID | Type | Training Data | Status |
|-------|---------------|------|--------------|--------|
| MT5-base-mMARCO | unicamp-dl/mt5-base-mmarco-v2 | Cross-encoder reranker | mMARCO (multilingual MS-MARCO) | ✅ Verified — used by the base paper |

### European Language Domain Models (Θ_D) — Base Paper's Models
| Model | HuggingFace ID | Language | Status |
|-------|---------------|----------|--------|
| mT5-base-german | Calizzano et al. (2022) | German | ✅ Used by base paper |
| mT5-base-french | Calizzano et al. (2022) | French | ✅ Used by base paper |
| mT5-base-spanish | Calizzano et al. (2022) | Spanish | ✅ Used by base paper |
| mT5-base-english | Calizzano et al. (2022) | English | ✅ Used by base paper |

### Indian Language Domain Models (Θ_D) — MT5 Family
| Model | Language | Status |
|-------|----------|--------|
| mT5-base-hindi | Hindi | ❌ NOT FOUND — no continued pretraining checkpoint exists |
| mT5-base-gujarati | Gujarati | ❌ NOT FOUND |
| mT5-base-bengali | Bengali | ❌ NOT FOUND |
| mT5-base-tamil | Tamil | ❌ NOT FOUND |

**Critical finding:** While the MT5-base family is the "safe" path for reproducing the base paper's European results, NO Indian-language continued-pretraining mT5-base checkpoints exist on HuggingFace. The European checkpoints exist only because Calizzano et al. (2022) specifically created them. Without Indian-language Θ_D checkpoints, the MT5 family cannot be directly used for Indian language task arithmetic.

**Implication:** This means the XLM-RoBERTa family — initially dismissed by Claude's deep research as "broken" — is actually the **primary viable path** for Indian language experiments. The MT5 family remains essential for reproducing the base paper's European results.

## 4.4 Family 3: mBERT (Encoder-Only, Secondary)

### Base Model (Θ₀)
| Model | HuggingFace ID | Parameters | Status |
|-------|---------------|------------|--------|
| mBERT | bert-base-multilingual-cased | ~178M | ✅ Verified |

### Indian Language Domain Models (Θ_D) — L3Cube Family
| Model | HuggingFace ID | Language | Base | Status |
|-------|---------------|----------|------|--------|
| HingMBERT | l3cube-pune/hing-mbert | Hinglish | mBERT | ✅ Verified |
| HindBERT v2 | l3cube-pune/hindi-bert-v2 | Hindi | MuRIL-base (NOT mBERT) | ⚠️ Base mismatch — uses MuRIL, not mBERT |
| MahaBERT v2 | l3cube-pune/marathi-bert-v2 | Marathi | MuRIL-base (NOT mBERT) | ⚠️ Base mismatch |
| DevBERT | l3cube-pune/hindi-marathi-dev-bert | Hindi+Marathi | MuRIL-base | ⚠️ Base mismatch |

### IR Models (Θ_T)
| Model | HuggingFace ID | Type | Status |
|-------|---------------|------|--------|
| mBERT MS-MARCO cross-encoder | ? | — | ❓ NEEDS VERIFICATION — no confirmed mBERT cross-encoder found |

**Note on MuRIL:** Google's MuRIL (google/muril-base-cased) is BERT-based but uses a different tokenizer and vocabulary than standard mBERT. L3Cube's HindBERT v2, MahaBERT v2, and DevBERT are fine-tuned FROM MuRIL, not FROM mBERT. This means their task vectors are compatible with MuRIL but NOT with mBERT. A MuRIL-based triplet would require a MuRIL MS-MARCO cross-encoder, which does not appear to exist.

## 4.5 Additional Θ_D Models (Various Architectures)

| Model | HuggingFace ID | Base Architecture | Language | Task Arithmetic Compatible? |
|-------|---------------|-------------------|----------|---------------------------|
| MuRIL | google/muril-base-cased | Modified BERT | 17 Indian languages | ❓ Need MuRIL-based IR model |
| IndicBERT v2 | ai4bharat/indic-bert | AlBERT-based | 12 Indian languages | ❌ Different architecture (AlBERT ≠ BERT/RoBERTa) |
| Indic-ColBERT | ai4bharat/indic-colbert | ColBERT architecture | 11 Indian languages | ❌ Different retrieval paradigm |

## 4.6 Model Landscape Summary

| Family | Θ₀ | Θ_T (Cross-Encoder) | Θ_D (Indian Languages) | Viable for Indian TA? |
|--------|-----|---------------------|----------------------|---------------------|
| **XLM-RoBERTa-base** | ✅ | ✅ (French-trained) | ✅ Hindi, Hinglish, Marathi; ❓ Gujarati, Bengali, Tamil | **YES — Primary path** |
| **MT5-base** | ✅ | ✅ | ❌ No Indian language checkpoints | **NO for Indian; YES for European reproduction** |
| **mBERT** | ✅ | ❓ Unverified | ✅ HingMBERT only (others use MuRIL base) | **Partial — only Hinglish** |
| **MuRIL** | ✅ | ❌ No IR model | ✅ HindBERT, MahaBERT, DevBERT | **NO — missing Θ_T** |

---

# PART V: DATASET LANDSCAPE

## 5.1 BEIR Benchmark Datasets (For Reproducing Base Paper)

| Dataset | Domain | Size | Available? | Source |
|---------|--------|------|-----------|--------|
| TREC-COVID | Pandemic biomedical | ~171K passages | ✅ Via beir/mteb library | BEIR benchmark |
| NFCorpus | Medical IR | ~3.6K docs | ✅ Via beir/mteb library | BEIR benchmark |
| SCIDOCS | Scientific citations | ~25K docs | ✅ Via beir/mteb library | BEIR benchmark |
| SciFact | Scientific fact-checking | ~5K docs | ✅ Via beir/mteb library | BEIR benchmark |

## 5.2 Indian Language IR Datasets

| Dataset | Languages | Qrels Type | Size | Available? | Citation |
|---------|-----------|-----------|------|-----------|----------|
| MIRACL Hindi | Hindi | Human-annotated ✅ | Standard | ✅ HuggingFace | Zhang et al., 2023 |
| MIRACL Bengali | Bengali | Human-annotated ✅ | Standard | ✅ HuggingFace | Zhang et al., 2023 |
| Hindi-BEIR | Hindi | 15 datasets, 7 tasks | Large | ✅ HuggingFace | Acharya et al., 2024 (arXiv 2409.05401) |
| mMARCO Hindi | Hindi | Machine-translated | ~8.8M passages | ✅ via ir_datasets | Bonifacio et al., 2021 |
| INDIC-MARCO | 11 Indian languages | Machine-translated | ~8.8M passages per language | ✅ HuggingFace (saifulhaq9/indicmarco) | Haq et al., ACL 2024 |
| MIRACL (other Indian) | — | — | — | ❌ Only Hindi & Bengali | — |

**Quality caveat:** MIRACL Hindi and Bengali have human-annotated relevance judgments (gold standard). INDIC-MARCO and mMARCO Hindi use machine translation (NLLB-1.3B-Distilled) — results on these carry a translation-quality caveat and are weaker evidence than native-qrels evaluations.

## 5.3 Code-Mixed IR Datasets

| Dataset | Language Pair | Size | Age | Available? | Notes |
|---------|--------------|------|-----|-----------|-------|
| FIRE MSIR 2014-2016 | Hindi (Devanagari + Roman) | ~66K docs, ~25 queries | Decade old | Archived on FIRE website | Outdated, very small, low MAP scores |
| CMIR 2025 (FIRE) | Bengali-English | 107,900 docs, 50 queries (20 train / 30 test) | Current | Via FIRE registration | Tiny test set; Bengali-English, NOT Hinglish |
| CS-MTEB / CSR-L | Multiple pairs | — | April 2026 | ❓ Need to verify release | From Zeng et al., ACL 2026 Findings |

**Critical gap:** No modern, large-scale Hinglish retrieval benchmark with proper qrels exists. This is the primary data risk for Direction C.

**Workaround options:**
1. Romanize Hindi-BEIR queries via transliteration (automated, synthetic — weaker evidence)
2. Use CMIR 2025 Bengali-English set (real but wrong language pair for Hinglish models)
3. Wait for CS-MTEB release from Zeng et al. (ACL 2026)

## 5.4 Multilingual Datasets (Base Paper Reproduction)

| Dataset | Language | Available? |
|---------|----------|-----------|
| GermanQuAD | German | ✅ |
| MIRACL Spanish | Spanish | ✅ |
| MIRACL French | French | ✅ |
| MIRACL English | English | ✅ |

---

# PART VI: IMPLEMENTATION TOOLS & COMPUTE

## 6.1 Model Merging Libraries

| Library | Methods Supported | Encoder Models? | Notes |
|---------|------------------|----------------|-------|
| **MergeKit** (arcee-ai/mergekit) | TIES, DARE, DARE-TIES, SLERP, Task Arithmetic, Linear, Model Soups | ✅ Architecture-agnostic (models must share dims) | Primary tool for our experiments |
| **HuggingFace PEFT** | TIES, DARE, weighted averaging (for LoRA adapters only) | ✅ | For adapter merging only, not full-weight |
| Custom implementation | Any | ✅ | Fallback: load state_dicts, subtract/add parameter-by-parameter |

## 6.2 Retrieval Pipeline Tools

| Component | Tool | Notes |
|-----------|------|-------|
| BM25 first-stage | Pyserini (Lucene-based) | Standard for BEIR evaluations; rank_bm25 for quick experiments |
| Cross-encoder scoring | sentence-transformers CrossEncoder | Base paper likely uses this |
| Evaluation metrics | pytrec_eval, ir_measures, BEIR library | P@10, NDCG@3/10, MAP@100 |
| Full BEIR evaluation | beir library or mteb library | Handles dataset loading + evaluation |

## 6.3 Compute Requirements

| Task | GPU | Estimated Time |
|------|-----|---------------|
| Task vector extraction (XLM-R-base, ~278M params) | Any GPU with ≥8GB VRAM | ~5 minutes |
| Cross-encoder re-ranking (100 docs × N queries) | RTX 6000 or H200 | Minutes to hours depending on corpus size |
| Merging (TIES/DARE/SLERP) | CPU sufficient for encoder models | Minutes |
| Full BEIR evaluation pipeline | RTX 6000 | Hours per dataset |

---

# PART VII: APPROVED RESEARCH DIRECTIONS

## 7.1 Direction B (CORE): Indian Language Transfer via Task Arithmetic

**What:** Apply the base paper's task arithmetic framework to Indian languages, testing whether language-specific task vectors from L3Cube models improve cross-encoder re-ranking for Hindi, Hinglish, Marathi (and potentially Gujarati, Bengali, Tamil if models are found).

**Why it's novel:** No prior work applies task arithmetic to Indian language IR.

**Model family:** XLM-RoBERTa-base (confirmed compatible triplets)

**Evaluation:** Hindi-BEIR, MIRACL Hindi, MIRACL Bengali, INDIC-MARCO

**RQ1:** Does task arithmetic transfer language knowledge effectively to Indian-language cross-encoder re-ranking, and how does performance vary across Indo-Aryan (Hindi, Bengali, Gujarati, Marathi) vs. Dravidian (Tamil) languages?

## 7.2 Direction A (METHODOLOGICAL EXTENSION): Advanced Merging Methods for IR

**What:** Systematically compare TIES-Merging, DARE, DARE-TIES, and SLERP against vanilla task arithmetic for IR re-ranking, evaluated on both the base paper's European datasets and Indian language datasets.

**Why it's novel (corrected framing):** Advanced merging has been applied to dense retrieval (Sasaki et al., CIKM 2025) but has NOT been systematically compared for multilingual cross-encoder re-ranking in typologically distant, low-resource settings.

**RQ2:** Do conflict-aware merging methods (TIES-Merging, DARE, SLERP) outperform vanilla task arithmetic for cross-lingual IR re-ranking on Indian languages, and which method best handles the subword fragmentation and script diversity unique to Indic settings?

## 7.3 Direction C (STRETCH GOAL): Code-Mixed / Hinglish Retrieval

**What:** Use HingRoBERTa (code-mixed pretrained model) task vectors for zero-shot code-mixed retrieval.

**Why it's novel:** Completely untouched intersection.

**Why it's a stretch goal, not core:** Evaluation data is stale/thin (no modern Hinglish IR benchmark). The cross-encoder Θ_T is French-trained (caveat). Implementation requires additional validation.

**Model triplet:** Confirmed compatible (XLM-R family).

## 7.4 Direction D (FUTURE WORK / ACKNOWLEDGED PRIOR ART): Multi-Vector Composition

**What:** τ_domain + τ_language simultaneously for IR.

**Why downgraded:** Pre-empted by Chronopoulou et al. (MRL 2024), Parović et al. (EACL 2024), and Klimaszewski et al. (COLING 2025). Only novel as "composition for IR re-ranking specifically," which is too narrow as a primary contribution.

---

# PART VIII: RECOMMENDED PROBLEM STATEMENT

## 8.1 Option 2 (Balanced) — Recommended

> Task arithmetic for IR (Braga et al., SIGIR 2025) uses only vanilla weighted addition and tests only European languages. While recent work has begun exploring model merging for dense retrieval (Sasaki et al., CIKM 2025), no study has systematically compared conflict-aware merging methods (TIES, DARE, SLERP) against vanilla task arithmetic for cross-encoder re-ranking in a multilingual setting — particularly for typologically distant, low-resource Indian languages with non-Latin scripts and rich morphology. This project provides the first controlled comparison of model-merging strategies for injecting Indian-language knowledge (Hindi, Marathi, and code-mixed Hinglish) into an XLM-RoBERTa MS-MARCO cross-encoder re-ranker, evaluated on MIRACL, Hindi-BEIR, and INDIC-MARCO with human-annotated and machine-translated relevance judgments. The contribution is the first systematic study of task arithmetic for Indian language IR, combined with the first evaluation of advanced merging methods for multilingual re-ranking.

## 8.2 Research Questions (2 core)

**RQ1:** Does task arithmetic transfer language knowledge effectively to Indian-language cross-encoder re-ranking, and how does performance vary across languages with different scripts and morphological complexity?

**RQ2:** Do conflict-aware merging methods (TIES-Merging, DARE, SLERP) outperform vanilla task arithmetic for cross-lingual IR re-ranking on Indian languages?

## 8.3 Fallback

If Direction A (advanced merging) shows no gains over vanilla TA, the project degrades gracefully to Direction B alone — a rigorous Indian language task arithmetic evaluation, which is still a novel and publishable contribution. Even "vanilla α-scaling is as good as TIES/DARE for cross-lingual re-ranking" is a clean, defensible negative result.

## 8.4 Target Venue

**Primary:** FIRE (Forum for Information Retrieval Evaluation) — aligns with Prof. Modha's IR specialization and the Indian language focus.  
**Secondary reach:** ECIR short paper, SIGIR short paper, or EMNLP Findings — if the merging comparison produces strong positive results.

---

# PART IX: THINGS TO FIND / OPEN QUESTIONS

## 9.1 Model Verification (Critical — Do Before Submitting Definition)

- [ ] **Verify L3Cube Gujarati RoBERTa exists** — search HuggingFace for l3cube-pune/gujarati-roberta or equivalent
- [ ] **Verify L3Cube Bengali RoBERTa exists** — search for l3cube-pune/bengali-roberta or equivalent
- [ ] **Verify L3Cube Tamil RoBERTa exists** — search for l3cube-pune/tamil-roberta or equivalent
- [ ] **Verify antoinelouis cross-encoder config.json** — confirm hidden_size=768, num_hidden_layers=12, num_attention_heads=12, vocab_size=250002 matches xlm-roberta-base exactly
- [ ] **Check if antoinelouis has other language versions** — mmarcoEN, mmarcoDE, etc. that might be better than French-only
- [ ] **Verify MuRIL architecture details** — confirm whether MuRIL is architecturally identical to mBERT (same hidden_size, layers, vocab) or uses different configuration
- [ ] **Search for mBERT MS-MARCO cross-encoder** — verify if any exists on HuggingFace
- [ ] **Search for mT5-base Indian language checkpoints** — continued pretraining (not task-specific fine-tuning) on Hindi/Gujarati/Bengali/Tamil

## 9.2 Dataset Verification

- [ ] **Download and test Hindi-BEIR** — verify format compatibility with beir/mteb libraries
- [ ] **Download MIRACL Hindi dev set** — verify qrels format and usability for α-tuning
- [ ] **Check CS-MTEB / CSR-L release status** — from Zeng et al. (ACL 2026 Findings, arXiv 2604.17632)
- [ ] **Check CMIR 2025 dataset access** — is FIRE registration required? Is the data downloadable?
- [ ] **Verify INDIC-MARCO** — download from saifulhaq9/indicmarco, check which languages have queries + passages + qrels

## 9.3 Implementation Verification

- [ ] **Clone base paper's codebase** — https://github.com/DetectiveMB/Task-Arithmetic-for-ZS-IR — verify dependencies, run one example
- [ ] **Test MergeKit on encoder models** — confirm TIES/DARE/SLERP work with XLM-RoBERTa-base family (not just LLaMA)
- [ ] **Test task vector extraction** — load xlm-roberta-base and l3cube-pune/hindi-roberta, subtract state_dicts, verify non-zero valid task vector is produced
- [ ] **Verify task vector injection into cross-encoder** — add task vector to antoinelouis cross-encoder's encoder layers only (not classification head), verify the merged model produces valid re-ranking scores

## 9.4 Literature Gaps Still to Verify

- [ ] **Read Sasaki et al. (CIKM 2025, arXiv 2509.21966) in full** — understand exactly which merging methods they applied to which retrieval models, and how our work differs
- [ ] **Read "Less Finetuning, Better Retrieval" (arXiv 2602.04731) in full** — understand their TIES/density sweep on BEIR
- [ ] **Read DRAMA (Kasela et al., arXiv 2602.14960)** — understand how the Braga group's follow-up differs from our approach
- [ ] **Read Kodali et al. (CODS 2025, arXiv 2510.19782)** — understand their task arithmetic + TIES for code-mixed classification
- [ ] **Read Zeng et al. (ACL 2026, arXiv 2604.17632)** — understand code-switching IR benchmark and results

## 9.5 Questions for Prof. Modha

- [ ] Is the scope (B + A, with C as stretch goal) appropriate for the MRP?
- [ ] Does he prefer a specific target venue (FIRE, ECIR, SIGIR short)?
- [ ] Would he be open to the XLM-RoBERTa family as primary (since MT5 lacks Indian checkpoints)?
- [ ] Is a negative result ("TIES/DARE don't help for Indian language re-ranking") acceptable as a deliverable?
- [ ] Should the project include a reproduction of the base paper's European results as a baseline?

---

# PART X: COMPLETE REFERENCE LIBRARY

## 10.1 Base Paper & Direct Predecessors

1. Marco Braga, Pranav Kasela, Alessandro Raganato, Gabriella Pasi. 2025. "Investigating Task Arithmetic for Zero-Shot Information Retrieval." SIGIR 2025. DOI: 10.1145/3726302.3730216. arXiv: 2505.00649.
2. Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, et al. 2023. "Editing Models with Task Arithmetic." ICLR 2023. arXiv: 2212.04089.
3. Pranav Kasela, Marco Braga, et al. 2026. "DRAMA: Domain Retrieval using Adaptive Module Allocation." arXiv: 2602.14960. (Follow-up from base paper authors)

## 10.2 Model Merging Methods

4. Prateek Yadav, Derek Tam, Leshem Choshen, Colin Raffel, Mohit Bansal. 2023. "TIES-Merging: Resolving Interference When Merging Models." NeurIPS 2023. arXiv: 2306.01708.
5. Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, Yongbin Li. 2024. "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch." ICML 2024. arXiv: 2311.03099.
6. Mitchell Wortsman et al. 2022. "Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time." ICML 2022. arXiv: 2203.05482.
7. Enneng Yang et al. 2024. "AdaMerging: Adaptive Model Merging for Multi-Task Learning." ICLR 2024. arXiv: 2310.02575.
8. Yan-Lun Chen et al. 2025. "Layer-Aware Task Arithmetic: Disentangling Task-Specific and Instruction-Following Knowledge." Findings EMNLP 2025. arXiv: 2502.20186.
9. Wenju Sun et al. 2025. "Task Arithmetic in Trust Region: A Training-Free Model Merging Approach to Navigate Knowledge Conflicts." ICLR 2025. arXiv: 2501.15065.

## 10.3 Advanced Merging Applied to IR (Counter-Examples to "Zero Papers" Claim)

10. Sasaki et al. 2025. "Effect of Model Merging in Domain-Specific Ad-hoc Retrieval." CIKM 2025. DOI: 10.1145/3746252.3760920. arXiv: 2509.21966.
11. "Less Finetuning, Better Retrieval." 2026. arXiv: 2602.04731.
12. "Temporal Information Retrieval via Time-Specifier Model Merging." 2025. arXiv: 2507.06782.
13. Hengran Zhang et al. 2026. "Bagging-Based Model Merging for Robust General Text Embeddings." arXiv: 2603.01161.

## 10.4 Cross-Lingual Model Merging

14. Marinela Parović, Ivan Vulić, Anna Korhonen. 2024. "Investigating the Potential of Task Arithmetic for Cross-Lingual Transfer." EACL 2024 Short Papers, pp. 124-137. DOI: 10.18653/v1/2024.eacl-short.12.
15. Shih-Cheng Huang et al. 2024. "Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages." ACL 2024.
16. Alexandra Chronopoulou et al. 2024. "Language and Task Arithmetic with Parameter-Efficient Layers for Zero-Shot Summarization." MRL@EMNLP 2024, pp. 114-126. DOI: 10.18653/v1/2024.mrl-1.7.
17. Lucas Bandarkar and Nanyun Peng. 2025. "The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in LLMs." MRL 2025. arXiv: 2505.18356.
18. Mateusz Klimaszewski et al. 2025. "No Train but Gain: Language Arithmetic for training-free Language Adapters enhancement." COLING 2025. arXiv: 2404.15737.
19. Yiran Zhao et al. 2025. "AdaMergeX: Cross-Lingual Transfer with Large Language Models via Adaptive Adapter Merging." NAACL 2025. DOI: 10.18653/v1/2025.naacl-long.493.
20. Emma Rafkin, Dan DeGenaro, Xiulin Yang. 2026. "Task Arithmetic with Support Languages for Low-Resource ASR." arXiv: 2601.07038.

## 10.5 Indian Language IR & NLP Resources

21. Arkadeep Acharya, Rudra Murthy, Vishwajeet Kumar, Jaydeep Sen. 2024. "Hindi-BEIR: A Large Scale Retrieval Benchmark in Hindi (Benchmarking and Building Zero-Shot Hindi Retrieval Model with Hindi-BEIR and NLLB-E5)." arXiv: 2409.05401.
22. Wasim (Saiful) Haq et al. 2024. "IndicIRSuite: Multilingual Dataset and Neural Information Models for Indian Languages." ACL 2024 Short Papers. DOI: 10.18653/v1/2024.acl-short.46. arXiv: 2312.09508.
23. Simran Khanuja et al. 2021. "MuRIL: Multilingual Representations for Indian Languages." google/muril-base-cased.
24. Ravindra Nayak and Raviraj Joshi. 2022. "L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models." WILDRE-6@LREC 2022. arXiv: 2204.08398.
25. Raviraj Joshi. 2023. "L3Cube-HindBERT and DevBERT: Pre-Trained BERT Transformer models for Devanagari based Hindi and Marathi Languages." ICICC 2023. arXiv: 2211.11418.
26. Raviraj Joshi. 2022. "L3Cube-MahaCorpus and MahaBERT: Marathi Monolingual Corpus." WILDRE. arXiv: 2202.01159.
27. Rémi Calizzano, Malte Ostendorff, Qian Ruan, Georg Rehm. 2022. "Generating Extended and Multilingual Summaries with Pre-trained Transformers." LREC 2022, pp. 1640-1650. (Source of mT5-base language-specific checkpoints used by base paper)

## 10.6 Code-Mixed / Code-Switched Retrieval

28. Qingcheng Zeng et al. 2026. "Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers." Findings ACL 2026. arXiv: 2604.17632.
29. Jinyeong Do et al. 2024. "ContrastiveMix: Overcoming Code-Mixing Dilemma." NAACL 2024 Short Papers. DOI: 10.18653/v1/2024.naacl-short.17.
30. S. Chanda, K. Tewari, S. Pal. 2025. "Findings of the Code-Mixed Information Retrieval from Social Media Data (CMIR) Shared Task at FIRE 2025." FIRE 2025 Working Notes. CEUR Workshop Proceedings.
31. Prashant Kodali et al. 2025. "Adapting Multilingual Models to Code-Mixed Tasks via Model Merging." CODS-COMAD 2025. arXiv: 2510.19782. (Task arithmetic + TIES for code-mixed classification, NOT IR)
32. Debajyoti Mazumder et al. 2026. "Neither Here Nor There: Cross-Lingual Representation Dynamics of Code-Mixed Text in Multilingual Encoders." Findings ACL 2026. arXiv: 2603.19771.

## 10.7 BEIR Benchmark & Evaluation

33. Nandan Thakur, Nils Reimers, Andreas Rücklé, et al. 2021. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021 Datasets Track.
34. Xinyu Zhang et al. 2023. "MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages." TACL 11, pp. 1114-1131.
35. Luiz Henrique Bonifacio et al. 2021. "mMARCO: A Multilingual Version of the MS MARCO Passage Ranking Dataset." arXiv: 2108.13897.

## 10.8 Retrieval Models Referenced

36. Rodrigo Nogueira et al. 2020. "Document Ranking with a Pretrained Sequence-to-Sequence Model (MonoT5)." Findings EMNLP 2020. DOI: 10.18653/v1/2020.findings-emnlp.63.
37. Longhui Zhang et al. 2023. "RankingGPT: Empowering Large Language Models in Text Ranking with Progressive Enhancement." arXiv: 2311.16720.
38. Nils Reimers and Iryna Gurevych. 2019. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.

---

*End of Report. Version 1.0. July 8, 2026.*  
*This document covers all research, analysis, and findings from the project initiation phase through deep research validation. The V2 Problem Statement will be produced as a separate document.*
