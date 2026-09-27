### 3.3. YAMNet Model

Yet another Audio Mobilenet Network, or in short the YAMNet model, is a pre-trained deep neural network incorporating the MobileNetV1 (depthwise separable CNN) architecture. This model was trained with the AudioSet ontology [32], and it can predict audio events from 521 classes from more than 2 million YouTube clips [33]. The dataset comprises various classes of environmental sounds, such as laughter, barking, sirens, etc.

### About the Model

the YAMNet model consists of 28 convolutional layers, 1 global average pooling layer, and 1 fully connected layer serving as its input and output layers. Depthwise separable convolutions and standard convolutions are sequentially stacked up to the pooling layer [33]. The convolutional layers in YAMNet utilize ReLU activation functions and incorporate a batch normalization technique [34]. Finally, the output layer utilizes a Softmax activation function to provide the sound class prediction [9].


### YAMNet-521

YAMNet-521 (the original Google model): Predicts 521 audio event classes from AudioSet, built on MobileNet-v1 depthwise-separable convolutions. The conv stack produces a 1024-dim embedding, which passes through a single logistic layer to yield 521 per-class scores per 960 ms input segment. ~3.2M parameters, expects a raw 16 kHz waveform (mel-spectrogram frontend baked in as custom layers). 
GitHub

### YAMNet-256
YAMNet-256 (ST's microcontroller variant): Because the default YAMNet is too large for most MCUs (>3M params), STMicroelectronics' model zoo provides a heavily downsized version that outputs 256-dim embeddings instead of 1024. The waveform→mel-spectrogram custom layers are stripped out (STEDGEAI can't convert them to C), so it takes 64×96 mel-spectrogram patches directly, and it's int8-quantized via the TFLite converter. ST trains Yamnet-256 on ESC-10 or a 5-class subset of FSD50K (knock, glass, gunshots, crying, speech) rather than the full 521-class AudioSet head — it's intended as a transfer-learning backbone with a small classifier on top. 


Hugging Face

### What is a log-mel spectrogram?

A log-mel spectrogram is a visual representation of an audio signal that maps frequencies to human perception (the Mel scale) and amplitudes to a decibel/logarithmic scale. It acts as a compact, 2D "fingerprint" of sound, making it the standard feature input for artificial intelligence models handling speech, music, and audio

### Things to Note

YAMNet's 521 outputs are independent per-class scores (multi-label, sigmoid-style), not a softmax — that's why several classes sit near 0.65 at once and the scores don't sum to 1.

## References

[https://pmc.ncbi.nlm.nih.gov/articles/PMC10347208/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10347208/)

### Glossary

AED stands for Acoustic Event Detection (sometimes also called Audio Event Detection