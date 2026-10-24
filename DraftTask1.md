# 1. Main text

## Conversational yielding beyond streaming speech generation

**Streaming speech generation is not sufficient for natural real-time conversation, because conversational control—deciding when to continue, hold, or yield while still listening—is not fully specified as a separate policy in these reports and is not measured by first-audio latency.**

This thesis concerns control over the conversational floor, rather than the ability to emit audio incrementally. Moshi explicitly supports simultaneous listening and speaking; its turn-taking is learned within joint stream modeling. Qwen2.5-Omni describes streaming perception and generation. These are different evidential claims. The research opportunity is to test whether a separately supervised controller can improve the distinction between a listener response and an attempt to take the floor, using information already available inside a frozen speech model.

**Evidence convention.** Paper-stated claims carry M or Q citations; architectural consequences are interpretation. The controller and all experiments are proposals. External related work is marked **[external]**. No experimental results are claimed. M denotes the attached Moshi paper; Q denotes the attached Qwen2.5-Omni Technical Report.

### 1. Seven architectural decisions and their conversational consequences

| Point | Moshi | Qwen2.5-Omni | Consequence for the thesis |
|---|---|---|---|
| **Encoder** | Causal SeaNet-based Mimi; WavLM semantic distillation (M §§3.3.1–3.3.2, pp.10–12). | Mel-spectrogram input; audio encoder initialized from Whisper-large-v3 (Q §2.2, p.4; §3, p.6). | Access to incoming speech does not itself specify an action on the floor. |
| **Representations** | Eight codebooks at 12.5 Hz; split semantic/acoustic quantization (M §§3.3.1–3.3.2, pp.11–12). | Input frames represent approximately 40 ms; discrete output uses qwen-tts-tokenizer (Q §§2.2–2.3, pp.4–5). | Representation quality and conversational decision quality require separate assessment. |
| **Alignment** | Summed token embeddings; aligned self-text supplies Inner Monologue (M §3.4.1, p.14; §3.4.4, p.16). | Adapters support alignment; Talker receives Thinker representations and sampled text embeddings (Q §3, p.6; §2.3, p.5). | Generation state may contain useful context beyond speech activity. |
| **Streaming** | Mimi uses 80 ms frames; reported latency is 160 ms theoretical and 200 ms practical (M §3.3.1, p.11; abstract, p.1). | Input attention uses 2-second blocks; DiT uses four blocks including two lookback and one lookahead (Q §2.4, p.5). | Processing delay and appropriate yielding are different quantities. |
| **Generation** | Temporal and Depth Transformers are autoregressive; Mimi decodes audio codes (M §3.4.1, pp.12–14; §3.3.1, p.10). | Talker is dual-track autoregressive; flow-matching DiT and BigVGAN produce waveform chunks (Q §§2.1–2.4, pp.3–5). | The intervention should target playback control without assuming a new generator. |
| **Speech completion** | Continuous audio represents speech or silence; PAD/EPAD organize self-text timing (M §3.4.4, pp.16–17). | Figure 2 shows text EOS; ChatML examples show message boundaries (Q Fig.2, p.3; §4.1, p.6). | Silence, text completion, and yielding cannot be treated as interchangeable signals. |
| **Training** | Weighted joint-token cross-entropy; conversation-specific fine-tuning stages (M §4.4, pp.20–21). | Staged multimodal training; Talker continuation, DPO, and speaker fine-tuning (Q §§3–4.3, pp.6–7). | Architectural capability and learned interaction behavior can have different causes. |

**Disclosure.** Qwen does not report output-codec rate/codebooks/bitrate, adapter dimensions/layers, output-block duration, numerical end-to-end latency, audio training hours/named audio corpora, or complete pretraining/tokenizer objectives. Neither report fully specifies input-endpoint, playback-completion/flush, or session-stop rules; Qwen additionally leaves interruption handling unspecified. Moshi’s practical-latency figure lacks a complete measurement protocol. Qwen’s “causal decoder” wording coexists with DiT lookahead (Q §§2.3–2.4, p.5). These omissions prevent matched codec, latency, data-efficiency, and stopping-protocol comparisons; they do not demonstrate performance failures.

### 2. Three limitations, ranked

#### 1. Continuous generation leaves conversational decisions implicit

**Addressed problem:** turn-segmented systems cannot naturally represent overlapping speech and interruptions. **Mechanism:** Moshi jointly models separate user and system streams, permitting simultaneous listening and speaking and natural silence (M §1, p.2; §3.4.4, p.17).

**Residual question:** when does that learned behavior correctly preserve or relinquish the floor? Moshi evaluates gaps, overlaps, and dialogue quality (M §5.6, p.31), but aggregate distributions cannot determine whether a particular backchannel caused an inappropriate stop. Qwen’s streaming architecture likewise does not establish this event-level behavior. The evidence therefore favors Moshi as the base for a yielding experiment: its simultaneous interaction is explicit. The unresolved claim is whether its internal state supports better action selection than an audio-trigger rule. Continuous modeling is a prerequisite for this experiment, not proof that its control decisions are adequate.

#### 2. Low transport delay does not establish timely social action

**Addressed problem:** serial processing and delayed audio generation obstruct rapid interaction. **Mechanisms:** Moshi uses causal codec frames and hierarchical prediction; Qwen uses block-wise processing and bounded-context waveform decoding (M §§3.3–3.4, pp.9–17; Q §2.4, p.5).

**Residual question:** does the system stop at the right moment after a genuine interruption while continuing through a backchannel? First-audio latency measures response onset, not that distinction. Faster yielding can be achieved trivially by suppressing output whenever speech appears; fewer false yields can be achieved by refusing to stop. Neither establishes improved conversation. Evaluation must compare false yields only under common interruption-latency and missed-interruption constraints. Input buffering, controller computation, and playback queues belong in that timing account. This is why a component-level timing diagram is more useful than comparing isolated headline latency figures.

#### 3. Available cues and training supervision may be insufficient

**Addressed problem:** speech models must combine linguistic content with acoustic detail. **Mechanisms:** Mimi distills semantic information while retaining acoustic codes; Qwen conditions Talker on semantic representations and discrete text to reduce phonetic ambiguity (M §3.3.2, p.12; Q §2.3, p.5).

**Residual question:** which information distinguishes acknowledgment, hesitation, and interruption in context? Speech activity identifies sound, not its conversational purpose. Generation state might indicate whether the assistant is completing a phrase, waiting, or continuing an explanation, but its usefulness must be tested. Training also matters: Moshi identifies shortcomings in simulated conversations without overlap, while Qwen acknowledges noisy speech supervision (M §4.4, pp.20–21; Q §4.3, p.7). An event-level improvement must therefore be attributed to a controlled state intervention, rather than differences between the models’ training corpora.

### 3. End-of-speech is four different decisions

| Decision | Operational meaning | Observable test |
|---|---|---|
| **Input endpoint** | The user has finished the current contribution. | Contrast a hesitation pause with a completed question. |
| **Conversational yield** | The assistant gives the user the floor. | Contrast a listener backchannel with a genuine interruption. |
| **Output completion** | The intended assistant response has finished playing. | Compare intended completion with actual playback and queue state. |
| **Session termination** | The conversation processor stops running. | Confirm this occurs only through the session’s explicit termination action. |

Moshi establishes continuous listening and generation, including silence; PAD/EPAD concern text timing rather than a standalone utterance-stop command (M §3.4.4, pp.16–17). Qwen establishes text/message boundaries and streaming waveform generation (Q Fig.2, p.3; §§2.3–2.4, p.5; §4.1, p.6). The disclosure above identifies the remaining specification boundary.

The test should preserve the floor through a user’s hesitation, allow an answer after a completed question, continue through a non-interrupting backchannel, and yield after an annotated genuine interruption. These are annotation targets, not assumed model behaviors. A hesitation during the user’s turn tests premature response onset; it cannot count as a false yield if the assistant has no active or pending output. Report that case separately. For hesitation episodes eligible for false-yield analysis, annotators must independently establish that the assistant should retain the floor.

### 4. Selected hypothesis

**We hypothesize that a frozen speech LLM’s own generation state improves continue/hold/yield decisions beyond incoming audio and observable playback history, which predicts fewer false yields on eligible backchannel and hesitation events than speech-activity rules and an audio-only learned controller, under matched interruption-to-yield latency and missed-interruption constraints.**

The hypothesis is refuted if the state-aware controller cannot improve that constrained comparison on held-out speakers and conversations, or if removing generation state preserves the benefit.

Four alternatives remain secondary: whether buffering dominates onset latency; whether codec phonetic quality predicts interaction accuracy; whether limited lookahead causes prosodic commitment errors; and whether real overlapping dialogue contributes beyond synthetic conversation training.

### 5. Proposed improvement: generation-state-aware yielding

#### Failure mechanism → proposed change

A speech-activity rule can respond to an acknowledgment as though it were a takeover attempt. This is a hypothesized baseline failure, not an established failure of either paper’s model.

Attach a small causal controller to **frozen Moshi**. Use the current Mimi input representation and causal speech-activity features, observable assistant playback activity, elapsed activity/silence durations, and queue occupancy. Add Moshi’s current Temporal Transformer context vector and self-text PAD probability as generation-state features; the architecture defines temporal context and text prediction (M §3.4.1, pp.13–14; §3.4.4, pp.16–17). Only values available at decision time may enter the controller.

A learned projection and recurrent classifier produce **continue, hold, or yield**. Continue releases available audio. Hold briefly suspends release while listening continues. Yield suppresses output, discards pending playback, and retains listening; the wrapper releases that state only after a new decision to resume. All actions share a bounded playback actuator. Hold duration is capped during validation, and inappropriate holding is counted as disruption rather than hidden as a successful non-yield.

The contribution under test is using the speech LLM’s generation state, not inventing turn-taking prediction. The base model’s parameters remain frozen throughout.

#### Training and labeled data

Train only the projection and classifier with supervised action cross-entropy. Labels must identify event type, floor ownership, event onset, acceptable action intervals, and interruption-response deadlines. Annotators should hear the surrounding dialogue and mark ambiguity independently of controller outputs. Future context can inform annotation but cannot become an input feature.

Collect consented human–Moshi conversations and controlled event reenactments, recording separate user and actually played assistant channels. Replay recordings through frozen Moshi to obtain causal feature traces. Fisher’s separate channels provide a potential auxiliary source for human conversational cues (M §4.2, p.18); new action annotations and human–system validation are still required. Split by speaker, conversation, and scenario, and retain disagreement scores for sensitivity analysis. Class weights and action thresholds are selected on development data; the test set remains untouched. Train on logged policy rollouts as well as controlled reenactments to expose the controller to its own action histories.

#### Predicted effect → baselines

The predicted effect is fewer inappropriate floor relinquishments without slower or less reliable interruption handling. Compare against a **speech-activity-triggered rule** that suppresses output after detected incoming activity satisfies an activity threshold and persistence condition. Include a **prespecified fixed-threshold version** and a **development-tuned version**. Give the tuned rule and learned controllers the same validation protocol and playback actuator; report unmodified Moshi as context.

A controller that only wins against the fixed rule has not established a generation-state benefit. A controller that wins only by accumulating additional output delay has not satisfied the hypothesis.

#### Primary experiment → primary metric

Use matched **backchannel, hesitation-pause, and genuine-interruption** scenarios, balancing speaker and recording conditions across systems. Annotate floor intent from context, rather than assigning intent from event duration alone. Completed questions are an endpoint sanity check outside the primary false-yield comparison.

The primary metric is the **unweighted mean of false-yield rates for eligible backchannels and hesitation episodes**, with both rates also reported separately. A false yield is suppression when floor-retention is annotated; an excessive hold crossing the annotated response tolerance counts as an equivalent disruption. Publish eligibility denominators, premature starts during user-owned hesitation, and time spent unnecessarily held.

Compare operating points under common limits on **interruption-to-yield latency** and **missed-interruption rate**. Set limits from development-set baseline behavior before testing. Measure latency from annotated interruption onset to actual playback cessation; count failure to yield within the annotated deadline as a miss. Report latency distributions with misses alongside them, not successful responses alone. If no controller operating point satisfies both constraints, the proposed benefit is unsupported. Estimate paired uncertainty by conversation, preserving within-dialogue dependence.

#### Diagnostic ablation → robustness check → controls

The diagnostic ablation removes Moshi generation-state features while retaining the acoustic features, playback history, queue state, controller capacity, and training procedure. It tests whether any benefit comes from a learned audio policy or knowledge that the assistant is speaking. Supply these observable cues to both learned variants; internal state must add information beyond them.

The robustness check repeats held-out event scenarios with controlled noise and echo perturbations. Keep the perturbation identical across variants and report both activity-detection and controller errors. A **Hindi/Telugu subset** asks whether cues learned elsewhere transfer, using fluent annotators and language-specific agreement reporting. Treat adaptation separately from zero-shot transfer; do not assume either base model fails on Indic speech.

Hold the Moshi checkpoint, generation settings, hardware, user recordings, playback mechanism, and annotation protocol fixed. Counterbalance live trials where replay cannot preserve conversational responses. This design does not control Moshi’s original training history or establish causality between Moshi and Qwen architectures.

#### How it could fail → alternative explanations

Generation state may encode timing or speaker identity rather than useful conversational context; observable-history controls and speaker-disjoint testing address those explanations. Threshold tuning may explain apparent gains; the tuned rule addresses that. Extra waiting may explain fewer false yields; latency constraints and hold accounting address it. Scripted cues may explain performance; natural held-out conversations test that dependence.

The wrapper also risks dialogue drift: Moshi can internally generate speech that playback suppression prevents the user from hearing. Freezing the model does not solve this mismatch. Track unheard output, resumption coherence, truncation, and controller overhead in follow-up turns. The proposal controls audible yielding; it does not claim to repair the generator’s internal dialogue state. Failure on these checks would limit or invalidate deployment despite a favorable primary metric.

### 6. Related work and contribution boundary

**dGSLM** supplies a duplex-dialogue comparison in Moshi’s evaluation (M §5.6, p.31); **Mini-Omni** motivates Qwen’s dual-track decoder (Q §2.1, p.3). **TurnGPT [external]** is a turn-taking prediction model ([Ekstedt and Skantze](https://aclanthology.org/2020.findings-emnlp.268/)), and **Voice Activity Projection [external]** is prior work on predicting conversational voice activity ([Ekstedt and Skantze](https://arxiv.org/abs/2205.09812)). These establish that turn prediction predates this proposal. The narrower question here is whether Moshi’s own generation state adds useful control information beyond matched audio and playback cues. Priority for that specific combination requires a wider literature check; this submission does not claim first use.

### 7. Position

Moshi is the justified experimental base because simultaneous listening and speaking are explicit in its formulation. Qwen shows why streaming input and output alone are insufficient evidence about event-level conversational decisions. The proposed controller tests a concrete extension of that distinction: whether internal generation context reduces false yields without delaying genuine interruptions. Its success depends on the constrained comparison, the state-removal ablation, and coherent behavior after playback suppression—not on a lower first-audio number.

---

# 2. Appendix

## Appendix A. Detailed architectural evidence

### A.1. Moshi: encoder and representation learning

Mimi uses a SeaNet-based causal audio autoencoder. Its encoder has four convolutional blocks with strides **(4, 5, 6, 8)** and a final convolution with stride **2**, mapping **24 kHz** mono audio to **512-dimensional** representations at **12.5 frames/s**. Transformer modules before and after quantization each have **8 layers and 8 heads**, with causal masking (M §3.3.1, p.10).

Mimi uses **8 codebooks**, each with **2,048 entries**, giving the reported **1.1 kbps** bitrate at **12.5 Hz** (M §3.3.1, p.11). Its split quantization combines one semantic VQ with a separate **7-level** acoustic RVQ; their outputs are summed for reconstruction (M §3.3.2, p.12).

WavLM provides **1,024-dimensional** teacher representations at **50 Hz**, pooled to **12.5 Hz**. Semantic distillation uses cosine distance. Non-causal teacher information is used during training while Mimi remains causal at inference (M §3.3.2, p.12). The paper evaluates an adversarial-only codec variant retaining feature and discriminator losses, alongside semantic distillation (M §3.3.1, p.11; §5.2, Table 4, p.24).

### A.2. Moshi: alignment, generation, and streaming

The Temporal Transformer receives summed learned embeddings across parallel token sequences. The Depth Transformer conditions on temporal context and preceding within-timestep tokens (M §3.4.1, pp.13–14). Inner Monologue aligns Moshi’s own text using Whisper word timestamps and PAD/EPAD tokens; user speech does not require an intermediate user transcript in this formulation (M §3.4.4, p.16).

The joint sequence contains both participants’ audio and Moshi’s text, with **K = 2Q + 1** and **Q = 8**. Actual user audio replaces predicted user-stream tokens during interaction (M §3.4.4, p.17).

Mimi’s initial frame and stride are **80 ms** (M §3.3.1, p.11). Moshi reports **160 ms theoretical** and **200 ms practical** latency (M abstract, p.1). Acoustic delay is **2** during pretraining and **1** during fine-tuning (M §5.3, p.27). Separate ASR/TTS configurations use **2 seconds** of text/audio delay; those settings are distinct from the conversational configuration (M §5.7, pp.31–32; Appendix C, p.56).

### A.3. Moshi: training requirements

The training sequence is Helium text pretraining, single-stream audio pretraining, simulated multi-stream post-training, Fisher fine-tuning, and synthetic instruction fine-tuning (M §4.4, pp.20–21).

Helium uses **2.1 trillion text tokens** (M §1, p.3). Text sources include CommonCrawl, Wikipedia-family sources, StackExchange, and peS2o (M §4.1, p.17). Audio pretraining uses **7 million hours**, predominantly English, transcribed with Whisper-large-v3. Fisher contributes **2,000 hours** of separately recorded conversation channels (M §4.2, p.18).

The synthetic instruction collection contains **more than 20,000 hours**. A separate **170-hour** conversation dataset supports TTS training and transcript-based preparation; Moshi is not directly trained on that audio collection (M §§4.2–4.3, p.18; §4.4, p.21). These figures should not be summed into a unique-data total.

Moshi’s joint-token objective uses cross-entropy with semantic-token weight **100** and acoustic-token weight **1** (M §4.4, p.21, Eq.7).

### A.4. Qwen2.5-Omni: input representation and alignment

Audio is resampled to **16 kHz** and transformed into a **128-channel** mel-spectrogram using a **25 ms** window and **10 ms** hop. Input representations correspond approximately to **40 ms** each (Q §2.2, p.4).

The report adopts the Qwen2-Audio encoder and initializes it from Whisper-large-v3. Adapters are trained before their encoders against the initially frozen LLM (Q §2.2, p.4; §3, p.6). The precise adapter position relative to positional encoding should be treated schematically in a reconstructed diagram.

Thinker is a Transformer decoder. Talker receives Thinker’s high-level representations and embeddings of sampled text tokens. The authors distinguish semantic similarity in hidden representations from phonetic specificity supplied by discrete text (Q §2.1, p.3; §2.3, p.5).

### A.5. Qwen2.5-Omni: generation and streaming

Talker autoregressively generates text and audio tokens using a dual-track decoder. Speech generation does not require word-level or timestamp-level alignment with text (Q §2.1, p.3; §2.3, p.5).

Input audio attention uses **2-second blocks**. Audio–video time interleaving also uses **2-second groups**, serving a separate organizational purpose (Q §2.2, p.4; §2.4, p.5).

The output path is speech codes → flow-matching DiT → mel-spectrogram → modified BigVGAN → waveform. DiT attends over **4 blocks**, including **2 lookback blocks** and **1 lookahead block** (Q §2.4, p.5). Flow-matching waveform synthesis should be distinguished from Talker’s autoregressive token generation.

### A.6. Qwen2.5-Omni: training requirements

Pretraining first freezes the LLM, then unfreezes all parameters, then extends sequence length. Second-phase additions comprise **800 billion image/video-related tokens**, **300 billion audio-related tokens**, and **100 billion video-with-audio-related tokens**. Context length increases from **8,192** to **32,768 tokens** (Q §3, p.6).

Thinker receives instruction fine-tuning across text, visual, audio, and mixed-modality conversations. Talker uses continuation next-token prediction, DPO, and multi-speaker instruction/speaker fine-tuning. Preference ranking uses WER and punctuation-pause error (Q §§4.2–4.3, pp.6–7).

## Appendix B. Experimental definitions and implementation checks

### B.1. Annotation and eligibility

Event class and desired action must be annotated separately. A brief utterance can be a takeover attempt; a longer acknowledgment can remain non-interrupting.

Annotators record:

- User and assistant activity intervals and floor ownership.
- Event onset and the earliest point where the intended action becomes inferable.
- Acceptable continue/hold/yield intervals and interruption deadlines.
- Whether assistant output is active or queued at event onset.
- Ambiguity and disagreement, including language-specific ambiguity.

Eligibility is determined before inspecting controller outcomes. User-owned hesitation pauses without assistant output contribute to the premature-start analysis, not the false-yield denominator.

### B.2. Metric accounting

For each eligible event category:

**False-yield rate = events with an inappropriate yield or equivalent excessive hold / eligible floor-retention events.**

The primary aggregate is the unweighted mean of the backchannel and hesitation rates. Report the category rates and denominators alongside it.

**Missed-interruption rate = genuine interruptions without timely playback cessation / genuine interruptions.**

Interruption-to-yield latency uses actual playback cessation, including queued audio and actuator delay. Report resumption errors separately: yielding correctly does not guarantee that subsequent speech is coherent.

### B.3. Replay versus interactive evaluation

For immediate action comparisons, start systems from a common assistant prefix and replay the same user event. This permits paired evaluation without attributing different user recordings to controller quality.

Replay does not reproduce how humans respond to different assistant actions. Follow-up live trials must therefore assess resumption and dialogue drift, with condition order counterbalanced. Treat replay and interactive outcomes separately.

### B.4. Controller state and integration

The classifier would use a causal recurrent history rather than future audio or future generated text. Feature extraction must be checked against the implementation’s actual execution order.

Holding may retain pending audio only within the configured queue and hold limits. Yielding clears pending playback. Neither action terminates listening or the session.

Before deployment, verify:

- Access to causal Mimi representations, temporal context, and PAD probabilities.
- Controller and feature-extraction overhead.
- Audible discontinuities after suppression and resumption.
- Divergence between internally generated and actually played speech.
- Whether the intervention preserves comprehension and subsequent response coherence.

“Lightweight” is a design target; parameter count, memory, and runtime must be measured.

### B.5. Hindi/Telugu evaluation

Use fluent annotators and naturally produced events rather than relying exclusively on translated scripts. Report annotation agreement and basic instruction comprehension separately from yielding behavior.

Evaluate transfer before language-specific adaptation. If adaptation is later performed, report it as a separate condition. Poor performance could reflect unfamiliar acoustic cues, lexical comprehension, annotation differences, or recording conditions; the design must not attribute it automatically to turn-taking control.

---

# 3. Figure specifications

## Figure 1. Architecture and timing dependencies

**Layout:** two horizontal model panels with a shared left-to-right time direction. Use arrows for dependencies, not proportional execution durations. Put citations in each panel caption.

### Panel A: Moshi

Draw these boxes and arrows:

**Incoming user waveform**  
→ **Mimi causal encoder + split quantization**  
→ **Learned token embeddings summed across streams**  
→ **Temporal Transformer: Helium-initialized**.

From the Temporal Transformer, draw:

- An arrow to **Inner Monologue text prediction**.
- An arrow to **Depth Transformer**.
- An arrow from **Inner Monologue text prediction** to **Depth Transformer**.

Continue:

**Depth Transformer**  
→ **Semantic + delayed acoustic codes**  
→ **Mimi decoder**  
→ **Playback queue**  
→ **Audible speech / natural silence**.

Add a return arrow from generated text/audio codes to **Previous joint history**, then back to the summed-embedding input.

Add a dashed training-only branch:

**WavLM teacher** → **Semantic distillation target** → **Semantic VQ**.

Timing labels:

- Beside Mimi input: **“80 ms initial frame/stride”**.
- Above the panel: **“Reported model latency: 160 ms theoretical; 200 ms practical.”**
- Under encoder compute, Transformer compute, decoder compute, and playback queue: **“Separate duration: not reported.”**
- Do not divide the reported total across the boxes.

Add the proposal as a visually separate dashed overlay:

**Incoming causal features** + **Temporal context / PAD probability**  
→ **Yielding controller: proposal**  
→ **Playback actuator: continue / hold / yield**.

The actuator controls the queue-to-speaker arrow. It must not interrupt the incoming-audio arrow.

**Caption sources:** M §§3.3–3.4, pp.9–17; abstract, p.1.

### Panel B: Qwen2.5-Omni

Draw:

**Incoming waveform**  
→ **Mel-spectrogram**  
→ **Audio encoder: block-wise attention**  
→ **Audio hidden representations / alignment**  
→ **Thinker**.

Put a note under alignment:

**“Adapters documented; exact ordering with positional encoding is schematic.”**

From Thinker:

- Arrow to **Sampled text tokens** → **Text response**.
- Arrow to **High-level hidden representations**.
- Arrow from sampled text to **Text-token embeddings**.

Then:

**Hidden representations + text embeddings**  
→ **Talker: dual-track autoregressive**  
→ **Speech codes**  
→ **Flow-matching DiT**  
→ **Mel chunks**  
→ **Modified BigVGAN**  
→ **Waveform playback**.

Timing labels:

- Input bracket: **“2-second attention blocks; not an end-to-end latency measurement.”**
- DiT bracket: **“4-block context: 2 past + current + 1 future.”**
- Under the future block: **“Duration: not reported.”**
- Under each compute stage and above total onset delay: **“Not reported.”**

**Caption sources:** Q §§2.1–2.4, pp.3–5; §3, p.6.

## Figure 2. Conversational events and desired actions

Draw four small panels. Each has aligned lanes:

1. **User audio**
2. **Assistant audible audio**
3. **Desired floor ownership**
4. **Desired controller action**

Use symbolic time without numerical durations.

- **Hesitation pause:** user speech → pause → user continuation. Floor ownership remains **user**; assistant remains silent. Label **“Endpoint control: do not start prematurely.”**
- **Backchannel:** assistant speech continues while user produces a short acknowledgment. Floor ownership remains **assistant**; label **“Continue; do not yield solely because speech is detected.”**
- **Genuine interruption:** user begins an annotated takeover during assistant speech. Draw assistant playback ending after onset. Mark the interval **“Interruption-to-yield latency: measured in experiment.”**
- **Completed question:** user finishes, then assistant begins. Mark **“Answer onset; distinct from interruption yielding.”**

Under the hesitation panel, add:

**“False-yield analysis includes only episodes with active/pending assistant output and an independently annotated assistant floor.”**

These are desired behaviors for the proposed test, not measured traces.

## Figure 3. Experimental design matrix

Draw this matrix:

| Condition | Incoming audio cues | Observable playback history | Moshi generation state | Threshold selection |
|---|---:|---:|---:|---|
| Unmodified Moshi | Native processing | Native processing | Native processing | No added controller |
| Fixed speech-activity rule | Yes | Same actuator interface | No | Prespecified |
| Tuned speech-activity rule | Yes | Same actuator interface | No | Development set |
| Audio-only learned controller | Yes | Yes | No | Development set |
| State-aware learned controller | Yes | Yes | Yes | Development set |

Place three event icons above it: **backchannel**, **hesitation**, **interruption**.

Below it, draw a decision gate:

**Satisfies common interruption latency and missed-interruption limits?**

- **No** → **Does not support proposed benefit**
- **Yes** → **Compare false-yield rates and uncertainty**

Add a side box:

**Held fixed:** checkpoint, generation settings, hardware, recordings, actuator, annotation protocol.

Add another:

**Separate checks:** noise/echo robustness; Hindi/Telugu transfer; resumption coherence.

No empirical curve is required. If a trade-off sketch is added, label it **“Hypothetical—no measurements.”**

---

# 4. VERIFY box

> **VERIFY before final submission**
>
> - **[external] TurnGPT:** confirm the claim that it is a turn-taking prediction model against [*TurnGPT: a Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog*](https://aclanthology.org/2020.findings-emnlp.268/).
> - **[external] Voice Activity Projection:** confirm the claim that it predicts conversational voice activity against [*Voice Activity Projection: Self-supervised Learning of Turn-taking Events*](https://arxiv.org/abs/2205.09812).
> - Before asserting novelty, check whether prior systems already use a speech generator’s hidden state for conversational action selection. The submission currently makes no priority claim.
> - Confirm implementation access to the proposed causal features and the behavior of Moshi’s state when internally generated audio is suppressed from playback.
> - Check Fisher access and permitted use before selecting it as an auxiliary annotation source.

---

# 5. Material not directly established by the two papers

The following content is interpretation or proposed design, not a reported result:

- **Thesis qualification:** “not fully specified as a separate policy” narrows your original wording so it does not deny Moshi’s implicit learned turn-taking.
- **Ranking the three limitations** and choosing Moshi as the experimental base.
- **The prediction that generation state adds useful conversational information**, and the selected hypothesis and refutation criteria.
- **The controller architecture:** feature selection, projection, recurrent classifier, action definitions, queue limits, and supervised training procedure.
- **Data collection and annotation plans**, including the proposed auxiliary use of Fisher for controller training.
- **Metric definitions, eligibility rules, matching constraints, baseline calibration, uncertainty estimation, and replay protocol.**
- **The internal-versus-audible history mismatch as an integration risk.** Its actual effect requires implementation checks.
- **Hindi/Telugu transfer, noise/echo robustness, and all expected failure modes.**
- **The external TurnGPT and Voice Activity Projection statements**, identified in the VERIFY box.
- **All figure layouts and desired conversational timelines.** They depict mechanisms or experimental targets, not measurements.

No gains, dataset sizes, controller parameter counts, runtime measurements, or experimental outcomes have been invented.