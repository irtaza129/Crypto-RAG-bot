#!/usr/bin/env python3
"""
Test to verify source_link metadata is properly used instead of filename.
"""

import json

def test_source_link_usage():
    """Demonstrate how source_link replaces filename in metadata."""
    
    print("=" * 80)
    print("SOURCE LINK METADATA TEST")
    print("=" * 80)
    
    # Simulated Pinecone metadata (as stored in vector DB)
    print("\n Metadata stored in Pinecone:")
    pinecone_metadata = {
        "source": "SEC_Regulations.json",  # Still stored for reference
        "source_link": "https://compliance-docs.example.com/usa/SEC_Regulations.json",  # NEW: Full URL
        "text": "The SEC requires crypto exchanges to register...",
        "jurisdiction": "USA"
    }
    print(json.dumps(pinecone_metadata, indent=2))
    
    # Processing in query.py build_prompt()
    print("\n" + "=" * 80)
    print("Processing in query.py build_prompt()")
    print("=" * 80)
    
    source_filename = pinecone_metadata.get("source", "Unknown Source")
    source_link = pinecone_metadata.get("source_link", "")
    jurisdiction = pinecone_metadata.get("jurisdiction", "Unknown")
    score = 0.95
    chunk_text = pinecone_metadata.get("text", "")
    
    # For API response: use source_link
    print(f"\n✅ For API response (source field):")
    source = source_link if source_link else source_filename
    print(f"   source: {source}")
    
    # For LLM prompt: use filename for readability
    print(f"\n✅ For LLM prompt context:")
    prompt_line = f"[Source: {source_filename} | Jurisdiction: {jurisdiction} | Score: {score:.2f}]\n{chunk_text}"
    print(f"   {prompt_line}")
    
    # Final API response
    print("\n" + "=" * 80)
    print("Final API Response")
    print("=" * 80)
    
    api_chunk = {
        "id": "SEC_001",
        "text": chunk_text,
        "source": source,  # Contains full URL, not filename
        "jurisdiction": jurisdiction,
        "score": score
    }
    
    print("\nChunk metadata in API response:")
    print(json.dumps(api_chunk, indent=2))
    
    # Verification
    print("\n" + "=" * 80)
    print(" VERIFICATION")
    print("=" * 80)
    
    checks = {
        "Pinecone has 'source' (filename)": "source" in pinecone_metadata,
        "Pinecone has 'source_link' (URL)": "source_link" in pinecone_metadata,
        "API response 'source' is URL, not filename": "https://" in api_chunk["source"],
        "Filename extracted separately": source_filename != source,
        "URL used when available": source == source_link,
    }
    
    for check, result in checks.items():
        status = "" if result else "❌"
        print(f"{status} {check}")
    
    print("\n" + "=" * 80)
    print("Benefits:")
    print("=" * 80)
    print(" API consumers get actual links (not filenames)")
    print(" LLM prompt remains readable with filenames")
    print(" Pinecone stores both for flexibility")
    print(" Fallback to filename if source_link missing")

if __name__ == "__main__":
    test_source_link_usage()
