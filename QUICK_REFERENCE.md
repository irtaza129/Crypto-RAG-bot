# Quick Reference: RAG Pipeline Metadata Implementation

## 🎯 What Changed

### Before
- Retrieved 3 chunks of text only
- No source or jurisdiction info in responses
- LLM had limited context

### After
- Retrieve 10+ chunks with full metadata
- Each chunk includes: id, text, source, jurisdiction, score
- LLM receives jurisdictions organized context
- API returns structured metadata

---

## 📊 Chunk Structure

```python
{
    "id": "SEC_001",                      # Unique identifier
    "text": "The SEC requires...",        # Chunk content (from Pinecone)
    "source": "SEC_Regulations.json",     # Source document
    "jurisdiction": "USA",                # Matched jurisdiction
    "score": 0.98                         # Relevance score (0-1)
}
```

---

## 🔄 Function Changes

### `retrieve_context(query, top_k=10)`
**Returns**: `results` dict with "matches" containing chunks with metadata

```python
# Example usage
results = retrieve_context("What are FCA rules?")
# results["matches"] = [{id, text, source, jurisdiction, score}, ...]
```

### `build_prompt(query, retrieved_docs)`
**Returns**: `(prompt_string, context_chunks_with_metadata_list)`

```python
# Example usage
prompt, chunks_with_meta = build_prompt(query, results)
# chunks_with_meta = [{id, text, source, jurisdiction, score}, ...]
```

### `rag_answer(query)`
**Returns**: `(answer_string, context_chunks_with_metadata)`

```python
# Example usage
answer, chunks = rag_answer("What are SEC rules?")
# chunks = [{id, text, source, jurisdiction, score}, ...]
for chunk in chunks:
    print(f"Source: {chunk['source']}, Jurisdiction: {chunk['jurisdiction']}")
```

---

## 🌐 API Response

### Before
```json
{
  "answer": "...",
  "retrieved_chunks": ["text1", "text2", "text3"]
}
```

### After
```json
{
  "answer": "...",
  "retrieved_chunks": [
    {
      "id": "SEC_001",
      "text": "The SEC requires...",
      "source": "SEC_Regulations.json",
      "jurisdiction": "USA",
      "score": 0.98
    },
    {
      "id": "FCA_001",
      "text": "The FCA requires...",
      "source": "FCA_Regulations.json",
      "jurisdiction": "UK",
      "score": 0.95
    }
  ]
}
```

---

## 🔍 Key Improvements

| Feature | Implementation |
|---------|-----------------|
| **Metadata Validation** | `retrieve_context()` checks all fields exist |
| **Jurisdiction Normalization** | `normalize_jurisdictions_with_llm()` handles variants |
| **Source Citation** | `build_prompt()` formats for LLM |
| **Chunk Count** | Default `top_k=10` instead of 3 |
| **API Types** | `ChunkMetadata` Pydantic model |

---

## 🚀 Usage Examples

### CLI Testing
```bash
python query.py

# Shows:
# [1] Source: SEC_Regulations.json | Jurisdiction: USA | Score: 0.95
#     The SEC requires crypto exchanges to register as MSBs...
# 
# [2] Source: FCA_Regulations.json | Jurisdiction: UK | Score: 0.92
#     The FCA treats crypto assets as financial instruments...
```

### API Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are FCA rules?"}'
```

### Streaming Query
```bash
curl -N -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are FCA rules?"}'
```

---

## 📝 Testing

Run verification tests:
```bash
# Test 1: Metadata structure
python test_metadata_retrieval.py

# Test 2: Integration pipeline
python test_integration_pipeline.py

# Test 3: Jurisdiction mapping
python test_jurisdiction_mapping.py
```

---

## ⚙️ Configuration

### Default Parameters
- **top_k**: 10 (retrieve at least 10 chunks)
- **candidates**: 80 (retrieve 80 to re-rank)
- **chunk_score_min**: 0.0 (accept all relevance scores)

### To Change
Edit `query.py`:
```python
# Line ~215: Change default top_k
def retrieve_context(query, top_k=10):  # Change 10 to desired value
```

---

## 🐛 Debugging

### Missing Metadata
If chunks lack metadata, check logs:
```
⚠️ Match {match.get('id')} missing 'text' in metadata
```

**Solution**: Re-upload data with `upload_json.py`

### Low Relevance Scores
If scores are < 0.7, query may not match documents well

**Solutions**:
1. Check if data is indexed
2. Use `enhance_query()` to improve search
3. Adjust jurisdiction detection

### No Jurisdictions Detected
Check if query contains jurisdiction keywords

**Examples**: USA, UK, SEC, FCA, MAS, Singapore, etc.

---

## 📚 Related Files

| File | Purpose |
|------|---------|
| `query.py` | Main RAG logic (retrieve, prompt, answer) |
| `api.py` | FastAPI endpoints |
| `upload_json.py` | Store metadata in Pinecone |
| `config/settings.py` | API keys and config |
| `utils/logger.py` | Logging utilities |

---

## 🔗 Data Flow

```
User Query
    ↓
Jurisdiction Detection (LLM + regex fallback)
    ↓
Retrieve Context (10+ chunks from Pinecone)
    ↓
Re-rank by jurisdiction match
    ↓
Build Enhanced Prompt (format with metadata)
    ↓
LLM Generation (cite sources)
    ↓
Return (answer + metadata chunks)
```

---

## 💡 Tips

1. **Frontend Integration**: Use `source` and `jurisdiction` fields to display chunk origins
2. **Sorting**: Use `score` field to show confidence ranking
3. **Filtering**: Filter chunks by `jurisdiction` on frontend if needed
4. **Caching**: Cache enhanced queries to speed up repeated requests
5. **Deduplication**: Check if multiple chunks from same source have same content

---

## ✅ Checklist for Using New Features

- [ ] Upgraded to latest version
- [ ] Ran all tests successfully
- [ ] Updated frontend to display metadata
- [ ] Tested API with metadata queries
- [ ] Verified sources display correctly
- [ ] Checked jurisdiction handling

---

**Last Updated**: December 11, 2025  
**Status**: Production Ready ✅
