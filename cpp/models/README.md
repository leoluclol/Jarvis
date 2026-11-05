# Wake-word model

`hey_jarvis.tflite` + `hey_jarvis.json` are the "Hey Jarvis" wake-word model
from [microWakeWord](https://github.com/kahrendt/microWakeWord) by Kevin
Ahrendt, distributed via the Apache-2.0 licensed `pymicro_wakeword` PyPI
package. They are copied here so the C++ build is self-contained.

- Author: Kevin Ahrendt (https://www.kevinahrendt.com/)
- License: Apache-2.0 (see `third_party/microfrontend/LICENSE`)
- Model input: `[1, 3, 40]` int8 (3 feature frames x 40 mel-filterbank bands)
- Model output: `[1, 1]` uint8 probability
