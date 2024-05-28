# Alive Models Server

A [PyTriton](https://triton-inference-server.github.io/pytriton) server of multiple models for different tasks like Automatic Speech Recognition, Speech Emotion Recognition, Facial Emotion Recognition, and Natural Language Processing (Sentiment Analysis)  
Small chunks of audio and text, as well as single images, can be sent to the server to get the results of the models.
The server can be deployed using Podman/Docker compose or Kubernetes and exposed to the internet using a reverse proxy like Nginx.  
The exposed REST and gRPC endpoints follow the [Predict Inference Protocol, version 2](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html).

## Demo

One or more of the models can be used to analyze audio, text, and images.
A short capture of an example that uses all the models can be seen below:

<video controls>
  <source src="demo.mp4" type="video/mp4">
</video>

## Models

The models that are currently being served are:

- ASR (Automatic Speech Recognition): [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- FER (Facial Emotion Recognition): [deepface](https://github.com/serengil/deepface) with [yolov8](https://github.com/ultralytics/ultralytics) for face detection.
- NLP (Natural Language Processing): [SamLowe/roberta-base-go_emotions-onnx](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx)
- SER (Speech Emotion Recognition): [hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0](https://huggingface.co/hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0)
