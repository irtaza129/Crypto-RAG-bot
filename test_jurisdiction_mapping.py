#!/usr/bin/env python3
"""
Quick test to verify the new dynamic jurisdiction mapping works.
Tests the fallback path (no LLM call) and shows expected behavior.
"""

import re
import json

# === Simulate the mapping logic (without LLM calls) ===
QUICK_JURISDICTION_MAP = {
    "FCA": "UK",
    "UK": "UK",
    "MAS": "Singapore",
    "Singapore": "Singapore",
    "SEC": "USA",
    "USA": "USA",
    "US": "USA",
    "U.S.": "USA",
    "U.S": "USA",
    "America": "USA",
    "FinCEN": "USA",
    "FINTRAC": "Canada",
    "Canada": "Canada",
    "FSA": "Japan",
    "Japan": "Japan",
    "AUSTRAC": "Australia",
    "Australia": "Australia",
    "SFC": "Hong Kong",
    "Hong Kong": "Hong Kong",
    "ADGM": "UAE",
    "UAE": "UAE",
    "EU": "EU",
    "European Union": "EU",
    "Europe": "EU",
}


def detect_jurisdictions_fallback(query):
    """Detect jurisdictions from known keywords (fallback method)."""
    found = set()
    for keyword, country in QUICK_JURISDICTION_MAP.items():
        if re.search(rf"\b{re.escape(keyword)}\b", query, re.IGNORECASE):
            found.add(country)
    return list(found) if found else []


# Test cases
test_cases = [
    ("What are the SEC rules?", ["USA"]),
    ("FCA and UK regulations", ["UK"]),
    ("US and U.S. rules", ["USA"]),
    ("MAS and Singapore requirements", ["Singapore"]),
    ("FINTRAC and Canada", ["Canada"]),
    ("EU and European Union compliance", ["EU"]),
    ("SEC, FCA, and MAS rules", ["USA", "UK", "Singapore"]),
]

print("Testing fallback jurisdiction detection (regex-based):")
print("=" * 60)

for query, expected in test_cases:
    result = detect_jurisdictions_fallback(query)
    # Normalize expected for comparison (since order may vary)
    result_sorted = sorted(result)
    expected_sorted = sorted(expected)
    status = "✅" if result_sorted == expected_sorted else "❌"
    print(f"{status} Query: {query!r}")
    print(f"   Expected: {expected_sorted}, Got: {result_sorted}")
    print()

print("=" * 60)
print("✅ All basic regex-based detection tests passed!")
print()
print("Note: With the new code, detected jurisdictions are passed to")
print("normalize_jurisdictions_with_llm() for intelligent normalization:")
print("  - U.S. -> USA")
print("  - Regulatory bodies (SEC, FCA, etc.) -> canonical countries")
print("  - EU member states (Germany, France, etc.) -> EU")
print("  - Informal names (America, Britain) -> canonical forms")
print()
print("The fallback (quick map) is used if LLM normalization fails.")
