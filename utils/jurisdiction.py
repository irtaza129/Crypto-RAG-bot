import re

CANONICAL_JURISDICTIONS = {
    # United States
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "sec": "United States",
    "fincen": "United States",

    # United Kingdom
    "uk": "United Kingdom",
    "fca": "United Kingdom",

    # European Union
    "eu": "European Union",
    "esma": "European Union",
    "europe": "European Union",

    # Singapore
    "singapore": "Singapore",
    "mas": "Singapore",

    # Others
    "japan": "Japan",
    "fsa": "Japan",
    "canada": "Canada",
    "fintrac": "Canada",
    "australia": "Australia",
    "austrac": "Australia",
    "hong kong": "Hong Kong",
    "sfc": "Hong Kong",
    "uae": "United Arab Emirates",
    "adgm": "United Arab Emirates",
}

GLOBAL_JURISDICTION = "Global"


def detect_jurisdictions(query: str) -> list[str]:
    """
    Detect jurisdictions from query.
    Always returns at least ['Global'].
    """
    found = []
    q = (query or "").lower()

    for key, value in CANONICAL_JURISDICTIONS.items():
        if re.search(rf"\b{re.escape(key)}\b", q):
            if value not in found:
                found.append(value)

    # Global is ALWAYS included as fallback
    if GLOBAL_JURISDICTION not in found:
        found.append(GLOBAL_JURISDICTION)

    return found
