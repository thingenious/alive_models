# API

For details on all the available endpoints, you can refer to the [standard inference protocols](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) and the [PyTriton](https://triton-inference-server.github.io/pytriton/latest/) docs.

## REST Examples

Some examples of how to use the REST API are shown below.

### ASR (Automatic Speech Recognition)

URL: `/v2/models/asr/versions/1/infer`  
Input Names:

- `data`: Base64 encoded audio data (of wav in 16-bit PCM format)
- `previous_transcript`: Previous transcription (can be an empty string)

Output Names:

- `text`: Transcription of the audio
- `segments`: Segments of the audio with timestamps and words

```python
import base64
import json
import httpx

headers = {"Content-Type": "application/json"}
previous_transcript = ""
audio_data = open("path/to/audio.wav", "rb").read()
b64_audio_data = base64.b64encode(audio_data).decode("utf-8")
request_data = json.dumps(
    {
        "id": "1",
        "inputs": [
            {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [b64_audio_data]},
            {"name": "previous_transcript", "shape": [1, 1], "datatype": "BYTES", "data": [previous_transcript]},
        ],
    }
)
url = "http://localhost:8000/v2/models/asr/versions/1/infer"
response = httpx.post(
    url,
    headers=headers,
    content=request_data,
    timeout=60,
)

print(response.json())

# Example Output:
# {
#     "id": "1",
#     "model_name": "asr",
#     "model_version": "1",
#     "outputs": [
#         {
#             "name": "text",
#             "datatype": "BYTES",
#             "shape": [1],
#             "data": ["hello world"]
#         },
#         {
#             "name": "segments",
#             "datatype": "BYTES",
#             "shape": [1],
#             "data": ["[{\"id\": 1, \"seek\": 10, \"start\": 0.0, \"end\": 1.0, \"text\": \"hello\", \"tokens\": [1, 2, 3, 4, 5], \"temperature\": 0.0, \"avg_logprob\": -0.1140624976158142, \"compression_ratio\": 1.0, \"no_speech_prob\": 0.0011053085327148438, \"words\": [{\"start\": 0.0, \"end\": 0.6, \"word\": \"Hello\", \"probability\": 0.96728515625}, {\"start\": 0.7, \"end\": 1.2, \"word\": \"World\", \"probability\": 0.99755859375]"]
#         }
#     ]
# }

```

### SER (Speech Emotion Recognition)

URL: `/v2/models/ser/versions/1/infer`

Input Names:

- `data`: Base64 encoded audio data (of wav in 16-bit PCM format)

Output Names:

- `label`: Emotion label
- `score`: Confidence score

```python
# ...
request_data = json.dumps(
    {
        "id": "16",
        "inputs": [
            {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [b64_audio_data]},
        ],
    }
)
url = "http://localhost:8000/v2/models/ser/versions/1/infer"
# ...
# Example Output:
#
# {
#     "id": "16"
#     "model_name": "ser",
#     "model_version": "1",
#     "outputs": [
#         {
#             "name": "label",
#             "datatype": "BYTES",
#             "shape": [1],
#             "data": ["happy"]
#         },
#         {
#             "name": "score",
#             "datatype": "FP32",
#             "shape": [],
#             "data": [0.8824779987335205]
#         }
#     ]
# }
```

### FER (Facial Emotion Recognition)

URL: `/v2/models/fer/versions/1/infer`

Input Names:

- `data`: Base64 encoded image data (of jpg/png format)

Output Names:

- `label`: Emotion label
- `score`: Confidence score

```python
# ...
image_data = open("path/to/image.jpg", "rb").read()
b64_image_data = base64.b64encode(image_data).decode("utf-8")
request_data = json.dumps(
    {
        "id": "1",
        "inputs": [
            {"name": "data", "shape": [1, 3, 224, 224], "datatype": "BYTES", "data": [b64_image_data]},
        ],
    }
)
url = "http://localhost:8000/v2/models/fer/versions/1/infer"
# ...
# Same output format as above (label and score)
```

### NLP (Natural Language Processing)

URL: `/v2/models/nlp/versions/1/infer`

Input Names:

- `text`: Text data

Output Names:

- `label`: Emotion label
- `score`: Confidence score

```python
# ...
text_data = "I am very happy today!"
request_data = json.dumps(
    {
        "id": "1",
        "inputs": [
            {"name": "text", "shape": [1, 1], "datatype": "BYTES", "data": [text_data]},
        ],
    }
)
url = "http://localhost:8000/v2/models/nlp/versions/1/infer"
# ...
# Same output format as above (label and score)
```
