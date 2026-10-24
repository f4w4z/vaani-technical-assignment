# Telugu speech pipeline

Python 3.10+; local Hugging Face Transformers inference. The four core files are
`config.yaml`, `asr.py`, `llm.py`, and `pipeline.py`.

```powershell
python -m pip install -r requirements.txt
python pipeline.py --audio "data/question.wav"
```

Use a real Telugu recording: no sample audio is bundled. WAV and FLAC are safe
choices; other formats depend on your installed libsndfile. First use downloads
the configured weights and needs internet access and disk space. Both models
remain loaded together. `config.yaml` defaults to `openai/whisper-small` and
`Qwen/Qwen2.5-1.5B-Instruct`, not the larger `whisper-large-v3` /
`Qwen2.5-7B-Instruct` pair, because the two must fit in VRAM simultaneously;
7 GB of free VRAM is not enough for both large checkpoints together (whisper-
large-v3 alone is ~3 GB in fp16, and Qwen2.5-7B-Instruct is ~15 GB). Swap in
the larger checkpoints if your hardware has headroom; `reports/task2.md`
records which pair the submitted evaluation run actually used.
CPU inference works but can be slow. Install the appropriate PyTorch GPU build
for your system before installing the other requirements.

Edit model names, revisions, device, precision, and decoding settings in YAML.
ASR accepts Transformers-compatible Whisper/IndicWhisper checkpoints. LLM
accepts causal instruction checkpoints with system/user chat templates; a base
model such as Sarvam-1 needs a model-specific prompt adapter. Model identifiers
or local model directories can be supplied; relative model directories resolve
from the working directory. Configured audio/output/cache paths resolve from the
YAML directory. CLI path overrides resolve from the current directory.

```python
from pipeline import SpeechPipeline, load_config

pipeline = SpeechPipeline.from_config(load_config("config.yaml"))
result = pipeline.run("data/question.wav")
print(result.transcript)
print(result.response)
print(result.asr_seconds, result.llm_seconds, result.total_seconds)
```

The command prints JSON and saves it at `paths.output` (UTF-8, Telugu preserved).
Results contain the audio path, transcript, response, and timings in seconds.
ASR timing includes local audio decoding/resampling; LLM timing includes prompt
encoding and output decoding. Accelerator work is synchronized. Model loading
is excluded. First inference may include kernel initialization: warm up each
model before comparing latency and pin dependency versions and model commits
for evaluation. Greedy decoding reduces variance but does not guarantee identical
results across hardware. Each pipeline instance is intended for sequential use.

These records are evaluation inputs, not accuracy scores. Telugu ASR accuracy
requires reference transcripts (e.g. WER/CER); response quality requires a labeled
evaluation set or explicit rubric. Telugu prompting does not guarantee language
quality. Validate both stages on your target recordings before deployment.

Run the lightweight checks without model downloads:

```powershell
python -m unittest discover -s tests -v
```

Model integration and real Telugu speech quality must be tested separately on
the target hardware. The project does not implement a vLLM backend.

## Batch evaluation

### Synthesize a Telugu evaluation corpus

No recorded Telugu corpus is bundled or committed. `synthesize_corpus.py`
builds one instead: it synthesizes ~130 template-generated Telugu factual
Q&A pairs (Indian state/UT capitals, and small addition problems spoken as
Telugu number words) to speech with the open `facebook/mms-tts-tel` model,
writing two parallel corpora in `prepare_data.py`'s sidecar format:

```powershell
python synthesize_corpus.py --clean-out corpus_clean --noisy-out corpus_noisy
```

`corpus_clean/` holds the synthesized audio as-is. `corpus_noisy/` holds the
same utterances with additive Gaussian noise at a fixed 8 dB SNR (`--snr-db`
to change it). These are the two "linguistic or acoustic conditions" and the
noisy corpus is also the bottleneck target for the research extension below.
Both are synthetic speech, not real recordings; `reports/task2.md` states
what that does and does not establish. Neither folder is committed (see
`.gitignore`) — regenerate them with this script, which is deterministic for
a fixed `--snr-db` given the same `transformers`/model revision.

### Prepare a local dataset

`prepare_data.py` creates the required JSON from a local folder or a CSV/TSV/JSON
manifest. It selects up to 100 samples by default with a reproducible random
seed, checks audio headers and labels, and retains Telugu in UTF-8. Audio files
stay in their source location; generated paths resolve from the output JSON.

For a folder, put a transcript and optional reference answer beside each audio:

```text
corpus/
  question_001.wav
  question_001.txt
  question_001.response.txt
```

Folder discovery recursively includes WAV, FLAC, and OGG files. Transcript files
contain only the spoken reference text, and response files contain a human-written
answer appropriate to that request. Missing/empty transcripts are errors.

```powershell
python prepare_data.py --source corpus --output data/test.json --limit 100 --seed 42
```

Manifests use `audio_path`, `ground_truth_transcript`, and
`ground_truth_response` columns by default; optional `id` values must be unique
strings. CSV/TSV headers and JSON object keys can be mapped:

```powershell
python prepare_data.py --source corpus/labels.csv --audio-column path --transcript-column text --response-column answer --output data/test.json
```

Manifest audio paths resolve from the manifest directory. A JSON manifest must
be an array of row objects. For a previously exported Telugu subset of FLEURS or
IndicVoices, map its local audio and transcript columns in the same way. This
utility works with local files; it does not download datasets or decode remote
dataset-specific audio objects. It preserves provided text and does not infer
whether a recording is actually Telugu, so select the Telugu subset upstream.

Speech corpora often contain read sentences rather than questions and have no
answer labels. For an explicit pipeline plumbing test, allow missing responses:

```powershell
python prepare_data.py --source corpus --output data/test.json --response-mode placeholder
```

Placeholder mode fills **only missing** responses with a fixed Telugu test
string. Each row includes `response_source` (`provided` or `placeholder`), which
the evaluator tolerates as an extra field. `test.metadata.json` records the
source, subset seed/count, and placeholder count. **Response quality measured
against placeholders is not a valid benchmark.** The evaluator does not
automatically filter these rows. Replace placeholders with reviewed answers
before interpreting response scores or error propagation. WER/CER still compare
the real reference transcripts. No answers are generated with the evaluated LLM.

Use `--limit 0` for every sample and `--overwrite` to replace existing prepared
output. Selection is repeatable for the same source ordering and seed. Validation
fails on missing/undecodable selected audio, duplicate audio/IDs, or empty labels;
it does not silently drop samples. Output metadata is separate from the JSON array.

### Run the evaluator

`evaluate.py` implements word/character Levenshtein distance from scratch,
measures latency and real-time factor, and compares each ASR-based answer with
an answer generated from the human transcript. Both are scored against the
same human-written ground-truth response. No evaluation library is required.

Create a UTF-8 JSON array with as many samples as needed (100+ is supported):

```json
[
  {
    "id": "te-001",
    "audio_path": "audio/question_001.wav",
    "ground_truth_transcript": "తెలంగాణ రాజధాని ఏది?",
    "ground_truth_response": "తెలంగాణ రాజధాని హైదరాబాద్."
  }
]
```

Audio paths resolve from the dataset JSON's directory. IDs are optional strings
(default: row number) and must be unique. References must be nonempty. Use actual
audio and labels; the example is a schema illustration, not a supplied dataset.

```powershell
python evaluate.py --dataset data/test.json --config config.yaml --output-dir outputs/evaluation --warmup 1
python -m unittest discover -s tests -p test_evaluate.py -v
```

The first command loads both models once and reuses them across all samples.
`--warmup 1` runs the first pair once without recording it, then evaluates every
sample including that first sample. Use `--warmup 0` to include first-inference
effects. Set `llm.generation.do_sample: false`; the CLI enforces this to reduce
generation variance in the paired comparison. Pin model revisions and dependency
versions and record hardware when comparing experiments.

Outputs in the selected directory:

- `report.json`: metadata, configuration, aggregate metrics, paired outputs,
  per-sample measurements, and failures.
- `samples.csv`: complete per-sample results.
- `failures.csv`: sample IDs, paths, and error details.
- `wer_vs_quality.csv`: plotting points; use `wer` on the x-axis and
  `asr_response_quality` on the y-axis. `clean_response_quality` and `quality_drop`
  are also supplied for paired plots. No plotting dependency is needed.

WER/CER and RTF are ratios, not percentages. WER/CER can exceed 1 because of
insertions. Corpus WER/CER divide total edit counts by total reference lengths;
the report also provides per-sample mean/median/p95 values. Text uses NFC Unicode
normalization and collapsed whitespace, preserving punctuation/case. CER ignores
whitespace and counts Unicode code points, including Telugu combining signs,
rather than grapheme clusters. Standalone metric functions return `None` for an
empty reference; dataset loading rejects empty ground-truth labels.

`rtf = total_seconds / audio_seconds` uses measured ASR-to-answer pipeline time,
including audio decoding/resampling and both model stages. It excludes model
loading, the additional clean-transcript LLM call, evaluation scoring, report
writing, and duration-metadata reads. `asr_rtf` is provided separately. Corpus
RTF is total pipeline seconds divided by total audio seconds. All pipeline
timings are wall-clock seconds with accelerator synchronization in the wrappers.
The evaluator reports mean, p50, and p95 stage/end-to-end latencies as well.

The default response score is **multiset word-overlap F1** in [0, 1]. It is a
transparent lexical proxy, not semantic correctness: valid Telugu paraphrases
or morphological variants may score poorly. Exact-match rates are also reported.
`quality_drop = clean_response_quality - asr_response_quality`; positive means
the ASR arm scored worse, and negative means it scored better. `degraded_fraction`
and `improved_fraction` summarize the paired signs. This comparison estimates
ASR's downstream effect under the chosen scorer and decoding conditions; it does
not establish that either generated answer is correct.

To use a task-specific scorer (reference first, generated answer second):

```python
from evaluate import evaluate_dataset, load_dataset, save_report
from pipeline import SpeechPipeline, load_config

pipeline = SpeechPipeline.from_config(load_config("config.yaml"))
samples = load_dataset("data/test.json")

def task_score(reference: str, hypothesis: str) -> float:
    # Replace with a validated rubric or domain-specific answer checker.
    return float(reference.strip() == hypothesis.strip())

report = evaluate_dataset(
    pipeline, samples, quality_scorer=task_score, quality_name="task_exact_match",
)
save_report(report, "outputs/task_evaluation")
```

Custom scores must be finite, higher-is-better values in [0, 1]. The same scorer
is applied to both arms. Programmatic callers must configure deterministic,
stateless generation themselves. The existing LLM wrapper retains no chat history.

By default, failed pairs (including empty-ASR failures) are recorded and excluded
from all aggregate metrics. Check `samples_failed` and `failure_rate` alongside
successful-only quality/latency: excluding failures can make results look better.
The CLI saves partial reports and exits with status 1 if any sample failed.
`--fail-fast` stops immediately instead, before final reports are saved.

### Plot evaluation results

Install the plotting dependency with `python -m pip install "matplotlib>=3.8,<4"`
(also included in `requirements.txt`). No pandas, seaborn, or model imports are
needed by `plot_results.py`.

```powershell
python plot_results.py --input outputs/evaluation/report.json --output-dir outputs/plots
# Full sample CSV also supports both plots:
python plot_results.py --input outputs/evaluation/samples.csv --output-dir outputs/csv_plots --quality-label "Word overlap F1"
# The plotting-points CSV supports the scatter only:
python plot_results.py --input outputs/evaluation/wer_vs_quality.csv --output-dir outputs/scatter
```

- `wer_vs_quality.png` shows ASR-transcript and human-transcript response scores
  against ASR WER. Both series use the same WER x-coordinate per sample. WER is
  a ratio and may exceed 1; no data is clipped at 100%.
- `latency_breakdown.png` stacks mean ASR time, mean LLM time, and remaining
  pipeline overhead. A separate bar shows the clean-transcript LLM latency when
  provided. That baseline call is excluded from the pipeline bar.

PNG files are rendered without opening a GUI, so the script also works on remote
machines. Outputs use the successful sample population. JSON plots include the
failed-sample count; CSV inputs cannot recover that metadata. Undefined scatter
values are omitted with per-series counts shown in the legend; malformed numeric
data and incomplete latency records are rejected. JSON metadata names the scorer;
use `--quality-label` to name it for CSV. Plotting accepts only evaluation outputs,
not the prepared dataset JSON. Replace placeholder responses before interpreting
quality plots. Existing plot files in the output directory are replaced.

### Correctness and hallucination rate

`hallucination.py` computes the assignment's "correctness and hallucination
rate" from saved `report.json` files without rerunning any model. For the
synthesized corpus each reference answer states one checkable fact (capital
name, or sum word); a response is `correct` if it contains that fact (sums
also accept the digit form), `abstained` if it instead contains an uncertainty
marker, and `hallucinated` otherwise. It is a lexical heuristic, not a
semantic judge; `reports/task2.md` §2.1 gives the full definition and results.

```powershell
python hallucination.py --reports outputs/evaluation/clean/report.json outputs/evaluation/noisy_baseline/report.json outputs/evaluation/noisy_denoised/report.json --output outputs/evaluation/hallucination.json
```

It relies on the `question_cap-*` / `question_sum-*` audio file names that
`synthesize_corpus.py` writes, so it only applies to that corpus.

### Confidence intervals

`bootstrap.py` gives a 95% paired bootstrap interval for the difference in
corpus WER or CER between two `report.json` files, resampling questions
(rows are paired by sample id, 10,000 resamples, fixed seed):

```powershell
python bootstrap.py --a outputs/evaluation/noisy_baseline/report.json --b outputs/evaluation/noisy_denoised/report.json --metric wer
```

An interval containing 0 means the difference is not distinguishable from
question-sampling noise. It does not capture speaker or recording variation
that the question set lacks.

### Unified report

`reports/report.pdf` is the single ACL-style paper that combines the
architecture analysis and the Telugu evaluation (source: `reports/report.tex`;
build with `pdflatex` twice from `reports/`). Its figures read the plots under
`outputs/plots/`, so regenerate those first. `reports/task1.*` and
`reports/task2.*` are the longer standalone write-ups of each part.

### Research extension: denoising for the noisy condition

`denoise.py` implements magnitude spectral subtraction (estimate a per-frequency
noise floor from each bin's low time-percentile magnitude, subtract it with a
floor to limit musical-noise artifacts, reuse the original phase on resynthesis).
It targets the measurable bottleneck the noisy condition exposes: additive noise
raises ASR WER relative to the clean condition. It is wired into `asr.py` behind
`asr.denoise` in the YAML config (default `false`, so existing runs are
unaffected) and included in measured ASR/total latency when enabled, since it
runs before transcription on every input, not only during evaluation.

```powershell
python evaluate.py --dataset data/test_noisy.json --config config.yaml --output-dir outputs/evaluation/noisy_baseline
python evaluate.py --dataset data/test_noisy.json --config config_denoised.yaml --output-dir outputs/evaluation/noisy_denoised
```

`config_denoised.yaml` is identical to `config.yaml` except `asr.denoise: true`.
Compare `outputs/evaluation/noisy_baseline/report.json` against
`outputs/evaluation/noisy_denoised/report.json` (and both against the clean-
condition run) for the baseline-vs-improved ablation; `reports/task2.md` reports
the actual numbers for this corpus and this noise level. `tests/test_denoise.py`
covers `spectral_subtract_denoise` directly, without downloading any model.

## Final verification and logging

```powershell
python verify_task2.py --config config.yaml --output-dir outputs/verification
```

`verify_task2.py` checks required source files and Python syntax, installed
dependency versions/imports against `requirements.txt`, configuration types and
ranges, Telugu transcription settings, deterministic decoding, and runtime device
availability. It creates/checks write access for the pipeline output directory,
model cache, verification directory, and default `data`, `outputs/evaluation` and
`outputs/plots` directories. Default evaluation/plot directories follow the current
working directory; configured paths follow the YAML directory. It does not create
or change your real recordings or install/downgrade any packages.

The script then generates exactly three silent dummy WAVs and runs the actual
`SpeechPipeline` and `evaluate_dataset` with injected ASR/LLM stand-ins. Known
perfect, substitution, and deletion cases are checked against fixed expected
WER/CER, response scores, signed quality drops, stage times, end-to-end times,
and RTF. Additional direct checks cover insertions, empty references, and Unicode
normalization. Both model outputs **and clocks/timings are synthetic**: this is
arithmetic and integration verification, not a Telugu accuracy or speed benchmark.
No model weights are loaded or downloaded, but the normal Python dependencies
must still be importable to load the existing pipeline module.

Outputs are regenerated on each invocation:

- `verification.log`: timestamped progress, check results, and error diagnostics.
- `checks.json`: structured PASS/FAIL/WARN/UNAVAILABLE statuses and independent
  setup-readiness / mock-pass indicators.
- `summary.md`: copyable Markdown tables of setup checks and the three dummy
  samples, plus corpus aggregates. Synthetic-results labeling stays with the table.
- `mock/`: dummy audio, evaluator-compatible `test.json`, and normal evaluator
  JSON/CSV reports, which can also be passed to `plot_results.py`.

Exit status is 0 when no FAIL/UNAVAILABLE checks remain; warnings (such as a
missing optional sample recording or unpinned model revisions) do not fail the
run. A missing/broken dependency or version outside the declared range causes
exit status 1. Other checks and the mock run are still attempted, and diagnostics
are saved whenever the verification output directory is writable. The Markdown
summary is printed as well as saved. The checks do not establish model checkpoint
availability, chat-template compatibility, model memory requirements, or real
speech/response quality; those require a labeled dataset and actual inference.

To plot the clearly synthetic verification output:

```powershell
python plot_results.py --input outputs/verification/mock/report.json --output-dir outputs/verification/mock/plots --quality-label "Mock word overlap F1"
```

References: [Whisper usage](https://huggingface.co/openai/whisper-large-v3),
[Qwen2.5 chat inference](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).
