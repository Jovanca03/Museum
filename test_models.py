
import ollama

# Test embedding
result = ollama.embed(
    model="bge-m3",
    input="Museum Cakraningrat menyimpan artefak bersejarah."
)

vectors = result["embeddings"]

print("Jumlah teks:", len(vectors))
print("Dimensi embedding:", len(vectors[0]))

# Test LLM
response = ollama.chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "user",
            "content": "Jelaskan secara singkat apa itu artefak museum."
        }
    ]
)

print("\nJawaban Qwen:")
print(response["message"]["content"])