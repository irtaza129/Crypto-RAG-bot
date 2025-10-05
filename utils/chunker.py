# Utility functions for text chunking

def chunk_text(text, chunk_size=1000, overlap=200):
	"""Split text into overlapping chunks."""
	chunks = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end]
		chunks.append(chunk)
		start += chunk_size - overlap
	return chunks
