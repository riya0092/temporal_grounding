# Temporal Knowledge Grounding System


## Overview

Built a system that automatically detects which queries need real-time data retrieval vs. which can be answered from knowledge base data. This addresses a core challenge in AI search: **temporal grounding** - knowing when AI-generated answers might be outdated.

## The Problem

AI systems trained on fixed data face a fundamental challenge:
- "What is Bitcoin trading at?" → Needs real-time data
- "Who invented the printing press?" → Knowledge base sufficient

How do we automatically detect this?

## Solution

End-to-end system combining:
- **Temporal Marker Detection** - Detects explicit ("current", "latest") and implicit ("price", "trading") time signals
- **Domain Volatility Taxonomy** - 6 domains with volatility scores (Finance=HIGH, History=STATIC, etc.)
- **Adaptive Scoring** - Combines signals based on query type

## Results

| Metric | Value |
|--------|-------|
| MAE (Freshness Prediction) | 0.20 |
| R² Score | 0.51 |
| F1 Score (Retrieval Decision) | 68% |
| Evaluation Dataset | 60 queries |

## Usage

```bash
pip install -r requirements.txt
python temporal_grounding.py

