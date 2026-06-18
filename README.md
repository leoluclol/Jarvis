# Jarvis: alexa ma con chatgpt

come funziona ora:
- input mic --> openwakeword (local) --> openai whisper + openai Gpt + openai tts (cloud) --> play audio



da fare:
- usare roba migliore di pyaudio che gracchia da bestia, potenzialmente usare mp3 invece di wav
- inserire conversazione fluente senza dover dire hey jarvis ogni volta
- inserire comando di stop
- forzare interruzione quando ci parlo sopra
- connettere accensione LED
- testare su raspberry Pi