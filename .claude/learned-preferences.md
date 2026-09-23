# Learned preferences

Corrections and confirmed approaches from working sessions, newest first.

## 2026-09-23: Docstrings state the math, not the reasoning

James: "your docstrings are way too chatty! less blathering more math."
A docstring gives the signature, the formula or contract, and the argument meanings.
Motivation, alternatives considered, and reassurance belong in a commit message or nowhere.
Example of the target register: `Replace each covariate column x by (x - mean(x)) / std(x), leaving the intercept alone.`
