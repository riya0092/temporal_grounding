# Temporal Knowledge Grounding System


## Overview

Built a system that automatically detects which queries need real-time data retrieval vs. which can be answered from knowledge base data. This addresses a core challenge in AI search: **temporal grounding** - knowing when AI-generated answers might be outdated.

## The Problem

AI models have a **timestamp problem**. They don't know:

- ❌ "Is this information still accurate?"
- ❌ "Does this query need current data?"
- ❌ "Should I retrieve new sources?"


**This project solves that.**

## What It Does

Given any query, the system outputs:

| Output | Description |
|--------|-------------|
| `score` | Freshness need (0.0 = stable, 1.0 = needs real-time) |
| `action` | `retrieve_fresh` vs `use_knowledge_base` |
| `domain` | Identified knowledge domain |

## Architecture

Query → Temporal Detector → Marker Score (0-1)
      → Domain Classifier → Volatility Score (0-1)
      → Combine (weighted by query type)
      → Decision (retrieve vs knowledge base)


## Temporal Marker Detection

| Marker Type | Examples | Impact |
|-------------|----------|--------|
| Current | now, today, right | +0.4 |
| Recent | latest, newest, most recent | +0.35 |
| Future | will, going to, predicted | +0.2 |
| Historical | was, ancient, invented | -0.3 |


## Domain Volatility 

| Domain | Volatility | Examples |
|--------|------------|----------|
| Finance | Critical (1.0) | Stocks, crypto, forex |
| News | Critical (0.9) | Breaking news, events |
| Technology | Medium (0.5) | Frameworks, tools |
| Science | Low (0.4) | Theories, constants |
| History | Static (0.0) | Historical facts |

## Results


Performance vs Baselines:

| Method | MAE | Improvement |
|--------|-----|-------------|
| Random (0-1) | 0.42 | baseline |
| Keyword Only | 0.23 | +45% |
| Full System | 0.22 | +49% |

Classification Metrics:

| Metric | Value |
|--------|-------|
| Accuracy | 84% |
| F1 Score | 90% |
| Precision | 82% |
| Recall | 100% |


 Dataset: 38 diverse queries across 6 domains

## What Works

- Finance/News queries correctly identified as high freshness need
- Historical/Static queries correctly identified as stable
-Combining temporal markers + domain volatility outperforms either alone

## Challenges

- "Current CEO" type queries remain challenging (rarely changes but must be accurate)

## Applications

- RAG Systems - Decide when to invoke retrieval
- Confidence Calibration - Adjust answer confidence based on freshness risk
- Compute Optimization - Skip expensive retrieval for stable queries
- User Transparency - Show "Based on data through [date]" disclaimers

## Installation 
```bash
pip install -r requirements.txt
python temporal_grounding.py

