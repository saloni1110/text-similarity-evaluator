#Taking manual input from user through terminal (two sentences)
from sentence_transformers import SentenceTransformer, util

# Load S-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Take sentences as input from the user
sentence1 = input("Enter the first sentence: ")
sentence2 = input("Enter the second sentence: ")

# Store them in a list
sentences = [sentence1, sentence2]
breakpoint()
# Generate sentence embeddings
embeddings = model.encode(sentences)

# Compute similarity
similarity = util.cos_sim(embeddings[0], embeddings[1])
print(f"S-BERT Cosine Similarity: {similarity.item():.4f}")
