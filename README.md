# Facial Recognition Training — Grove AI Cam

> A self-improving face recognition pipeline with bootstrap-aware consensus learning. The more it recognises you, the more confidently it learns.

---

## Concept

Standard face recognition requires a manually curated training dataset. This system builds its own.

When the camera detects a face it tentatively recognises, it captures a frame and adds it to the encoding pool. As the pool grows, a consensus mechanism kicks in — the system only learns from frames where a majority of existing encodings agree. The recognition improves continuously in a closed, controlled environment.

**The feedback loop:**
```
Detect face → Check distance to known encodings
    → Bootstrap phase: learn freely (first 10 encodings)
    → Consensus phase: only learn if 40%+ of encodings agree
    → Save frame + encoding → Recognition improves → Threshold tightens
```

---

## Hardware

- **Grove AI Camera** — edge inference camera module
- Standard webcam also supported (OpenCV)

---

## How It Works

### Bootstrap Phase
For the first `BOOTSTRAP_MIN` (default: 10) encodings, the system learns from any face that passes the distance threshold. This seeds the encoding pool without requiring manual setup.

### Consensus Phase
Once bootstrapped, a new frame is only learned if `CONSENSUS_RATIO` (default: 40%) of existing encodings vote for it. This prevents drift and stops the model from learning bad frames.

### Temporal Stability
A face must remain stable in frame for `TEMPORAL_TIME` (default: 0.3s) before a frame is captured. This filters out blurry or transitional frames.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `RECOG_THRESHOLD` | 0.45 | Distance threshold for display label |
| `LEARN_THRESHOLD` | 0.42 | Distance threshold for learning |
| `TEMPORAL_TIME` | 0.3s | Stability window before capture |
| `CONSENSUS_RATIO` | 0.40 | Minimum vote ratio for consensus learning |
| `BOOTSTRAP_MIN` | 10 | Encodings before consensus activates |
| `MAX_SESSION_LEARN` | 20 | Auto-stop after this many learned frames |

---

## Getting Started

```bash
pip install face_recognition opencv-python numpy
```

### Bootstrap a new face

```bash
python quick_learn.py
```

Sit in front of the camera. The system will automatically capture frames as it recognises you. Learning stops at `MAX_SESSION_LEARN` frames per session.

### Run recognition only

```bash
python test_only.py
```

### Continuous learning with temporal tracking

```bash
python recognize_temporal_learn.py
```

---

## Files

```
├── quick_learn.py                # Bootstrap-aware session learning
├── recognize_temporal_learn.py   # Continuous recognition + learning
├── test_only.py                  # Recognition without learning
├── encodings.npy                 # Saved face encodings (generated)
└── names.npy                     # Corresponding identity labels (generated)
```

---

## Limitations & Known Issues

- Single-identity trained (currently hardcoded to `"yuu"` target — generalise for multi-person use)
- No anti-spoofing (photo attacks not defended against)
- Encoding drift possible over very long sessions without periodic validation

---

## Roadmap

- [ ] Multi-person support
- [ ] Anti-spoofing layer (liveness detection)
- [ ] Grove AI Cam edge inference integration (on-device)
- [ ] Integration with MIKA as an identity verification module

---

**Ayush Birhmaan · Aerospace Engineering, Amity University**
