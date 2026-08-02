import openwakeword

# Download 'hey_jarvis' and the required Google speech feature embeddings
openwakeword.utils.download_models(model_names=["hey_jarvis"])

print("Successfully downloaded 'Hey Jarvis' model!")