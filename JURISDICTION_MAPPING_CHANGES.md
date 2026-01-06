# Dynamic Jurisdiction Mapping - Implementation Summary

## Overview
Enhanced `query.py` to use **dynamic LLM-based jurisdiction normalization** that intelligently maps various jurisdiction references to canonical country names.

## Changes Made

### 1. Renamed Static Map (Line 27)
- **Old**: `JURISDICTION_MAP`
- **New**: `QUICK_JURISDICTION_MAP`
- **Why**: Now used as a fallback; LLM is the primary mapper

### 2. Added `detect_jurisdictions_fallback()` (Lines 57-64)
- Regex-based fallback detection from `QUICK_JURISDICTION_MAP`
- Used when LLM normalization is unavailable

### 3. Added `normalize_jurisdictions_with_llm()` (Lines 67-136)
**Core new function** that intelligently maps:

#### Handles:
- **Spelling variants**: U.S., US, U.S → USA
- **Regulatory bodies**:
  - SEC, FinCEN, OCC → USA
  - FCA, PRA → UK
  - ESMA, EBA → EU
  - MAS → Singapore
  - FSA → Japan
  - AUSTRAC → Australia
  - SFC → Hong Kong
  - FINTRAC → Canada
  - ADGM → UAE
- **EU member states**: Germany, France, Italy, Spain, Netherlands, Poland, etc. → EU
- **Regional references**: Europe, European → EU
- **Canonical names**: Passed through unchanged (USA, UK, EU, etc.)

#### Implementation:
1. Takes raw detected jurisdictions as input
2. Sends to Gemini with detailed normalization rules
3. Returns normalized list (removes duplicates, preserves order)
4. **Fallback**: If LLM fails, uses `QUICK_JURISDICTION_MAP` for each raw jurisdiction

### 4. Updated `detect_jurisdictions()` (Lines 139-145)
- Now calls `detect_jurisdictions_fallback()` first (regex)
- Then pipes results through `normalize_jurisdictions_with_llm()` (LLM)
- Returns normalized jurisdictions

### 5. Updated `check_crypto_relevance_and_jurisdiction()` (Lines 157-162)
- Extracts raw jurisdictions from LLM response
- Immediately normalizes them via `normalize_jurisdictions_with_llm()`
- Returns clean, canonical jurisdiction list

## Benefits

✅ **Handles variant spellings** (U.S., US, USA all → USA)
✅ **Maps regulatory bodies to jurisdictions** (SEC → USA, FCA → UK, MAS → Singapore)
✅ **Normalizes EU member states** (Germany, France, etc. → EU)
✅ **Graceful fallback** if LLM is unavailable (quick map still works)
✅ **Reduces false positives** in retrieval (no duplicate jurisdictions)
✅ **Better retrieval prioritization** with clean, normalized jurisdiction tags

## Example Flow

```
User Query: "What are the U.S. and FCA rules for crypto exchanges?"
          ↓
detect_jurisdictions_fallback()
          ↓ (regex match)
["USA", "UK"]
          ↓
normalize_jurisdictions_with_llm(["USA", "UK"])
          ↓ (LLM enrichment)
["USA", "UK"]  (returned normalized)
          ↓
retrieve_context() prioritizes documents tagged with USA, UK
```

## Testing

Run the included test file to verify fallback logic:
```bash
python test_jurisdiction_mapping.py
```

Expected output: All 7 test cases pass (including multi-jurisdiction queries).

## Notes

- LLM normalization is done via Gemini API (requires `GEMINI_API_KEY`)
- If LLM fails (network error, API issue), falls back to `QUICK_JURISDICTION_MAP`
- Deduplicates results (no duplicate jurisdictions in response)
- Maintains order of first occurrence
- Works seamlessly with existing `retrieve_context()` and `rag_answer()` functions
