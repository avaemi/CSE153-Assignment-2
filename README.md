
# Assignment 2 — Music Generation

## Overview
This project explores two approaches to automatic music generation using machine learning.

Task 1 trains an unconditioned model on the GiantMIDI Piano dataset to learn a distribution over melodies and sample new ones. 
Task 2 trains a bidirectional LSTM on the JSB Chorales dataset to harmonize a given soprano melody — predicting the accompanying alto, tenor, and bass voices at each 16th-note timestep. 
Both tasks include exploratory data analysis, a baseline comparison, full training pipeline, and generated MIDI output.

## Video Presentation
https://youtu.be/GcoAZAE0vbw

## Tasks
- **Task 1:** Symbolic Unconditioned Generation
- **Task 2:** Symbolic Conditioned Generation — Bach Chorale Harmonization (melody → chords)

## Dataset
- **Task 1:** GiantMIDI Piano
- **Task 2:** JSB Chorales (Boulanger-Lewandowski et al., 2012) — 382 four-part Bach chorales at 16th-note resolution

## Files
- `assignment2.ipynb` — main notebook
- `workbook.html` — exported notebook for submission
- `video_url.txt` — YouTube link
- `task1/` — Task 1 code, data, and outputs
- `task2/` — Task 2 code, data, and outputs
- `symbolic_unconditioned.mid` — Task 1 generated MIDI
- `symbolic_conditioned.mid` — Task 2 generated MIDI

