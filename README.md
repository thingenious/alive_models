# Alive Models Server

A [PyTriton](https://triton-inference-server.github.io/pytriton) server of multiple models for different tasks like Automatic Speech Recognition, Speech Emotion Recognition, Facial Emotion Recognition, and Natural Language Processing (Sentiment Analysis)  
Small chunks of audio and text, as well as single images, can be sent to the server to get the results of the models.
The server can be deployed using Podman/Docker compose or Kubernetes and exposed to the internet using a reverse proxy like Nginx.  
The exposed REST and gRPC endpoints follow the [Triton Inference Server API](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/customization_guide/inference_protocols.html).

## Examples

One or more of the models can be used to analyze audio, text, and images.  
You can find examples of each case in the [./examples](examples) directory.
A short capture of an example that uses all the models can be seen below:

<video src="https://github.com/thingenious/alive_models/assets/4764837/12b60c17-c8f1-4aaf-9097-4ceacf8f4e98" type="video/mp4"></video>

## Models

The default models to be served are:

- ASR (Automatic Speech Recognition): [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- SER (Speech Emotion Recognition): [hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0](https://huggingface.co/hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0)
- FER (Facial Emotion Recognition): [deepface](https://github.com/serengil/deepface) with [yolov8](https://github.com/ultralytics/ultralytics) for face detection.
- NLP (Natural Language Processing): [SamLowe/roberta-base-go_emotions-onnx](https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx)
- LID (Language Identification): [cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid)
- TTS (Text-to-Speech): [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)

## Deployment

- Podman/Docker compose: [deploy/compose/README.md](deploy/compose/README.md)
- Kubernetes: [deploy/k8s/README.md](deploy/k8s/README.md)

## License

[MIT](LICENSE)
