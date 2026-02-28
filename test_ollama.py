import ollama

response = ollama.chat(
    model='phi:latest',
    messages=[
        {'role': 'user', 'content': 'Say hello in a friendly way'}
    ]
)

print(response['message']['content'])