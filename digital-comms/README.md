# End-to-End Digital Communication System Simulator

A complete Python-based simulation of a digital communication system — from analog source to bit detection — built from scratch using NumPy and Matplotlib.

## Overview

This project implements and validates the full transmitter-receiver chain of a digital communication system, including PCM encoding, line coding, pulse shaping, AWGN channel modelling, matched filtering, and symbol detection.

**Validated over 1,000,000 bits. Simulated BER matches theoretical benchmarks.**

---

## System Architecture

```
Analog Source
     ↓
PCM Sampling & 4-bit Quantization (fs = 1000 Hz, 16 levels)
     ↓
Line Coding (5 schemes: NRZ, Unipolar, AMI, RZ, Manchester)
     ↓
Pulse Shaping (Raised Cosine, β = 0, 0.25, 0.5, 1.0 | Sinc)
     ↓
AWGN Channel (SNR range: −4 dB to +12 dB)
     ↓
Matched Filter (maximises SNR at receiver)
     ↓
Symbol Detection & BER Computation
```

---

## Key Results

| SNR (dB) | Simulated BER | Theoretical BER |
|---|---|---|
| −4 | 3.21 × 10⁻² | ~3.5 × 10⁻² |
| −2 | 1.23 × 10⁻² | ~1.4 × 10⁻² |
| 0  | 3.57 × 10⁻³ | ~4.0 × 10⁻³ |
| 2  | 7.15 × 10⁻⁴ | ~8.0 × 10⁻⁴ |
| 4  | 8.30 × 10⁻⁵ | ~9.0 × 10⁻⁵ |
| 6  | 4.00 × 10⁻⁶ | ~4.5 × 10⁻⁶ |
| 8+ | 0 (error-free) | ~0 |

Simulated BER closely matches theoretical Q-function predictions, validating the correctness of the implementation.

---

## Features Implemented

**Transmitter:**
- Analog signal generation (sinusoidal, 50 Hz)
- PCM sampling and 4-bit uniform quantization
- 5 line-coding schemes: Polar NRZ, Unipolar NRZ, Bipolar AMI, Polar RZ, Manchester
- Pulse shaping: Raised Cosine (β = 0, 0.25, 0.5, 1.0) and Sinc

**Channel:**
- AWGN noise addition at configurable SNR

**Receiver:**
- Matched filter (cross-correlation with pulse shape)
- Symbol sampling and threshold detection
- BER computation against transmitted bits

**Analysis & Visualisation:**
- Eye diagrams for all pulse shapes
- BER vs SNR waterfall curve (simulated vs theoretical)
- Line-code waveform comparison
- Raised-cosine time/frequency domain plots

---

## Observations

**Pulse Shaping Trade-off:**
- NRZ/RZ: sharp transitions, minimal ISI, higher bandwidth
- Raised Cosine (β=1.0): cleanest eye opening, reduced ISI, higher bandwidth usage
- Raised Cosine (β=0): sinc-equivalent, maximum spectral efficiency, impractical due to infinite duration
- Higher β → cleaner eye, more bandwidth consumed

**Matched Filter Effect:**
- At 4 dB SNR, eye diagram nearly closed after AWGN
- After matched filter: eye reopens significantly, confirming SNR maximisation at decision point

---

## Tech Stack

```
Language:    Python 3.x
Libraries:   NumPy · Matplotlib · SciPy (erfc for theoretical BER)
Concepts:    PCM · Line Coding · Pulse Shaping · AWGN · Matched Filtering
             ISI · Eye Diagrams · BER Analysis · Signal Detection
```

---

## How to Run

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run full simulation
python digital_comms_system.py
```

Generates all plots sequentially. BER waterfall curve (1M bits) takes ~30 seconds.

---

## Project Structure

```
digital-comms/
├── digital_comms_system.py   # Complete simulation code
├── results/                  # Output plots
│   ├── pcm_quantization.png
│   ├── line_codes.png
│   ├── eye_diagrams.png
│   ├── raised_cosine.png
│   └── ber_waterfall.png
└── README.md
```

---

*B.Tech ECE Project — IIIT Delhi | Ayushee Kaul (2024147)*
