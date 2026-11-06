# FastSpeech 2 for English (Indian-accented) and Telugu

Two non-autoregressive TTS models trained from scratch on **~5 h of speech each**, built on the
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) implementation of
[FastSpeech 2 (Ren et al., 2021)](https://arxiv.org/abs/2006.04558):

| | English | Telugu |
|---|---|---|
| Corpus | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) — Indian-accented English, crowd-sourced, 117 speakers | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) — studio, 1 male + 1 female |
| Train / val (selected audio) | 4.97 h / 1.13 h — 3,064 / 671 clips, 115 speakers in both | 5.0 h / 1.0 h — 2,168 / 434 clips, 2 speakers |
| Train / val (speech after silence trimming) | 4.65 h / 1.06 h | 4.19 h / 0.85 h |
| Text frontend | normaliser (times, dates, numbers) + g2p_en (CMUdict, POS-based heteronyms, neural OOV) | **new**: rule-based Telugu G2P + Telugu number/date/time normaliser |
| Alignment | MFA, pretrained `english_us_arpa` + corpus lexicon from the same G2P | MFA acoustic model **trained on the corpus** (no pretrained Telugu model exists; recipe without SAT) |
| Vocoder | HiFi-GAN (universal, pretrained) | HiFi-GAN (universal, pretrained) |

Results: audio for every test sentence, duration / F0 / energy plots of the variance adaptor,
loss curves, and an ASR-based intelligibility evaluation (CER/WER of synthesised vs. recorded
speech) are in [`Results/`](Results/).


## Results

Both models: 100k steps, batch 16, checkpoint at step 100k (chosen by listening, see below).
Training took 7.4 h for English (sharing the GPU) and 2.2 h for Telugu (alone, 12.7 it/s) on one RTX 6000 Ada.

### Intelligibility (ASR on 60 random validation sentences + the 5 test sentences)

ASR: `openai/whisper-large-v3` (English), `vasista22/whisper-telugu-large-v2` (Telugu).
"Real recordings" is the ASR error on the original validation audio, i.e. the floor; "synthesis" is
FastSpeech 2 + HiFi-GAN speaking the same sentences with the same speaker.

| CER / WER (%) | English | Telugu |
|---|---|---|
| Real recordings (val) | 2.67 / 5.48 | 4.39 / 28.86 |
| **Synthesis (val)** | **9.24 / 16.20** | **6.46 / 33.60** |
| CER gap (synthesis ÷ real) | 3.5× | 1.5× |
| Synthesis (test sentences) | 11.48 / 25.81 | 3.14 / 19.51 |

CER is the meaningful metric for Telugu: it is agglutinative, so long words make WER harsh even on
real recordings (28.9 %). Per-utterance transcripts are in `Results/<Lang>/asr_eval.json`.

### Observations

- **Data quality beats data quantity per speaker.** Telugu (2 studio speakers, ~2.5 h each) loses
  far less intelligibility relative to real speech than English (115 crowd-sourced phone recordings,
  ~2.5 min per speaker): a 1.5× vs 3.5× CER gap with the same model, steps and amount of audio.
- **English prosody does not generalise at 2.5 min/speaker.** Validation pitch and duration losses
  plateau after ~1k steps (pitch ≈ 0.42, duration ≈ 0.23) while the training losses fall to ≈ 0.01:
  the model learns each speaker's mean pitch and rate quickly, then memorises clip-level prosody.
  Duration targets are also noisy, because a US-English aligner was applied to Indian-accented speech.
- **Validation mel loss is a poor model-selection criterion.** Its minimum (English: step 10k)
  sounded clearly worse than 50k–100k. The loss uses ground-truth durations and pitch, so it never
  scores the model's own prosody. Final checkpoints were picked by listening; `val_best.pth.tar` is kept.
- **Heteronyms work through the frontend:** both "conduct" sentences (noun E2a, verb E2b) are
  transcribed perfectly (0 % CER); g2p_en's POS-based lookup gives `K AA1 N D AH0 K T` vs
  `K AA0 N D AH1 K T`. Telugu పట్టు (grip / silk, T2a/T2b) is a true homograph with identical
  pronunciation, so identical output is the correct behaviour.
- **Very long rare words are the main English failure:** E1 "antidisestablishmentarianism"
  (22 % CER, heard as "van dyck's establishment in his…") — a neural-G2P guess plus long-run
  duration drift. The equally long Telugu loanword న్యూరోఫార్మకోలాజికల్ (T1) is read correctly,
  since Telugu spelling is phonemic and the rule-based G2P is exact.
- **Telugu forced alignment needed a different MFA recipe.** MFA parallelises per speaker, so
  speaker-adaptive training on a 2-speaker corpus ran on ~2 cores at >1 h per iteration. Dropping
  the SAT stages (monophone → triphone → LDA) aligned 2,602 / 2,620 clips in 17 minutes.

---

## How it works

```
text ─► normalise ─► G2P ─► phones ─► Encoder ─► Variance adaptor ─► Decoder ─► mel ─► HiFi-GAN ─► wav
                                                  ├ duration predictor ─► length regulator
                                                  ├ pitch predictor  (phoneme-level F0)
                                                  └ energy predictor (phoneme-level energy)
```

Training targets for the variance adaptor come from the data: phoneme durations from Montreal
Forced Aligner, F0 from WORLD (dio + stonemask) averaged per phoneme, energy = L2 norm of the
STFT magnitude averaged per phoneme. Pitch and energy are z-normalised (statistics from the
training split only) and quantised into 256 bins whose embeddings are added to the hidden sequence.

### Data selection (same code for both languages — [`fs2/data_prep.py`](fs2/data_prep.py))

1. **Hard filters:** 1–15 s duration, ≥ 2 words, plausible speaking rate (chars/s of the
   normalised text), < 0.1 % clipped samples, transcript in a script the frontend can speak.
2. **Quality score:** a frame-energy SNR proxy (95th − 10th percentile of frame RMS in dB). Svarah is
   crowd-sourced, so clips below 12 dB are dropped and cleaner clips are preferred within each speaker.
3. **Speaker-stratified budget:** each speaker gets a share of the 6 h proportional to its usable
   audio (capped), takes its highest-quality clips, and splits them 5 : 1 into train/val. Every
   validation speaker is therefore seen in training (required by the speaker embedding), but no
   clip is in both sets.
4. **Transcript disjointness:** a val clip whose text also occurs in train is swapped for an
   unused clip with unseen text when possible, so validation loss doesn't partly measure memorised text.
5. Clips matching any test sentence are excluded.

Every run writes `data/<Lang>/preprocessed/data_report.json` (hours, speakers, per-speaker minutes,
filter counts, overlap checks) and `preprocess_report.json` (alignment drops).

### Telugu frontend ([`Telugu/frontend.py`](Telugu/frontend.py))

Telugu script is an abugida with near-phonemic spelling, so the G2P is deterministic:
every consonant carries an inherent /a/ unless a vowel sign or the virama (్) follows;
the virama yields conjuncts and geminates (పట్టు → `p a tt tt u`); the anusvara (ం) becomes the
homorganic nasal of the next consonant (బలంగా → `b a l a ng g aa`, ఉంచండి → `u ny c a nn dd i`)
and /m/ word-finally (సమావేశం → `… sh a m`); Telugu keeps the word-final vowel (no schwa
deletion as in Hindi). The normaliser expands `2026` → రెండు వేల ఇరవై ఆరు,
`12వ` → పన్నెండవ, `10:45` → పది నలభై ఐదు.

### Changes to the reference implementation

| Area | Change |
|---|---|
| **Bug fix** | Variance adaptor applied the *pitch* control factor to the energy predictor (`e_control` was ignored). |
| **Bug fix** | MFA ≥ 2 writes silences as *empty* intervals, and `tgt` drops these by default, which shifts every duration after a pause. TextGrids are now read with `include_empty_intervals=True`. |
| Data leakage | Pitch/energy normalisation statistics are fitted on the training split only (the reference fits them on everything). |
| Split | Speaker-stratified, duration-targeted 5 h / 1 h split decided upstream (the reference uses a random `val_size` cut). |
| Languages | Per-language symbol tables (ARPAbet with stress for English; new Telugu phone set) instead of one mixed table; new Telugu G2P and normaliser; English normaliser for times and dates. |
| Portability | Runs on current PyTorch / librosa: removed hard-coded `.cuda()` in the STFT, fixed librosa ≥ 0.10 API calls, added `weights_only=False` for torch ≥ 2.6, device taken from tensors. |
| Training | One `config.yaml` per language; auto-resume; CSV loss logs; val-best checkpoint kept for reference (final model chosen by listening, see below); vectorised length regulator; multiprocess feature extraction. |
| Evaluation | Variance-adaptor plots per test sentence; ASR-based CER/WER evaluation. |

Unchanged from the reference: Transformer encoder/decoder (FFT blocks), variance predictors,
PostNet, loss (L1 mel + MSE on log-duration, pitch, energy), Noam-style LR schedule, HiFi-GAN.

---

## Repository layout

```
English/  train.py  preprocess.py  inference.py  config.yaml  frontend.py  checkpoints/
Telugu/   train.py  preprocess.py  inference.py  config.yaml  frontend.py  checkpoints/
fs2/      shared FastSpeech 2 core (model, transformer, variance adaptor, data prep,
          feature extraction, trainer, synthesis + plots, HiFi-GAN)
scripts/  setup_vast.sh  01_data.sh  02_train.sh  03_results.sh
          plot_losses.py  evaluate_asr.py  export_checkpoints.py  smoke_test.py
test_sentences/  english.tsv  telugu.tsv
Results/<Lang>/  training_loss.png  validation_loss.png  test_generated_audios/
                 variance_adaptor_plots/{duration,pitch,energy}/
Report/
```

## Reproducing

Hardware used: 1× RTX 6000 Ada (48 GB), ~10 vCPU, 50 GB RAM (vast.ai). Data prep ≈ 2 h, training 2–7 h per model.

```bash
git clone <this repo> && cd <repo>
bash scripts/setup_vast.sh            # pip deps, NLTK data, MFA in a micromamba env
export HF_TOKEN=hf_...                # accept the Svarah terms on Hugging Face first
bash scripts/01_data.sh               # both languages in parallel: download → filter/split → MFA → features
bash scripts/02_train.sh              # trains English + Telugu in the background (resumable)
bash scripts/03_results.sh            # audio, plots, loss curves, ASR eval, slim checkpoints
```

Individual steps:

```bash
python English/preprocess.py --stage prepare     # → data/English/raw, lexicon.txt, split.json, data_report.json
MFA_CMD="micromamba run -n aligner mfa" \
python English/preprocess.py --stage align       # → data/English/TextGrid
python English/preprocess.py --stage features    # → data/English/preprocessed
python English/train.py                          # → English/checkpoints, logs/English
python English/inference.py                      # → Results/English
python English/inference.py --text "Any sentence, 5:30 PM." --id demo --speaker <name>
python scripts/plot_losses.py English
python scripts/evaluate_asr.py English --n_val 60
```

`Telugu/` takes the same commands. Inference only needs the exported checkpoint folder:
`<Lang>/checkpoints/` holds `fastspeech2_<lang>.pth.tar` (Git LFS), `stats.json` and `speakers.json`.

Run `python scripts/smoke_test.py english|telugu` to test the whole pipeline on CPU with synthetic audio
(no dataset, GPU or MFA needed).

## Credits

Model code adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT, see
`LICENSE_ming024_FastSpeech2`); HiFi-GAN from [jik876/hifi-gan](https://github.com/jik876/hifi-gan).
Data: Svarah (AI4Bharat, CC BY 4.0), IndicTTS Telugu (IIT Madras, Indic TTS license).
