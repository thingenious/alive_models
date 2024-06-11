# API

For details on all the available endpoints, you can refer to the [standard inference protocols](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md) and the [PyTriton](https://triton-inference-server.github.io/pytriton/latest/) docs.

## REST Examples

Some examples of how to use the REST API are shown below.

### ASR (Automatic Speech Recognition)

URL: `/v2/models/asr/versions/1/infer`  
Input Names:

- `data`: Base64 encoded audio data (of wav in 16-bit PCM format)

Output Names:

- `results`: JSON dumped segments of the audio with their text, the timestamps and words

```python
import base64
import json
import httpx

headers = {"Content-Type": "application/json"}
audio_data = open("path/to/chunk.wav", "rb").read()
b64_audio_data = base64.b64encode(audio_data).decode("utf-8")
request_data = json.dumps(
    {
        "id": "1",
        "inputs": [
            {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [b64_audio_data]},
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

# Example output:
# {
#     "id": "1",
#     "model_name": "asr",
#     "model_version": "1",
#     "outputs": [
#         {
#             "name": "results",
#             "datatype": "BYTES",
#             "shape": [],
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

- `results`: JSON dumped predictions with the `label` and `score` for each emotion

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
# Example output:
# {
#   "id": "134",
#   "model_name": "ser",
#   "model_version": "1",
#   "outputs": [
#     {
#       "name": "results",
#       "datatype": "BYTES",
#       "shape": [],
#       "data": [
#         "[{\"score\": 0.9841493964195251, \"label\": \"happy\"}, {\"score\": 0.012710321694612503, \"label\": \"disgust\"}, {\"score\": 0.0015516174025833607, \"label\": \"angry\"}, {\"score\": 0.0011115961242467165, \"label\": \"fear\"}, {\"score\": 0.0002755543973762542, \"label\": \"surprise\"}]"
#       ]
#     }
#   ]
# }
```

### FER (Facial Emotion Recognition)

URL: `/v2/models/fer/versions/1/infer`

Input Names:

- `data`: Base64 encoded image data (of jpg/png format)

Output Names:

- `results`: JSON dumped predictions with the `label` and `score` for each emotion

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
# Example output:
# {
#   "id": "108",
#   "model_name": "fer",
#   "model_version": "1",
#   "outputs": [
#     {
#       "name": "results",
#       "datatype": "BYTES",
#       "shape": [],
#       "data": [
#         "[{\"label\": \"angry\", \"score\": 9.990368239735993e-07}, {\"label\": \"disgust\", \"score\": 3.2478467246928735e-12}, {\"label\": \"fear\", \"score\": 1.9463645189950467e-07}, {\"label\": \"happy\", \"score\": 99.98987317024906}, {\"label\": \"sad\", \"score\": 2.089022626165558e-05}, {\"label\": \"surprise\", \"score\": 1.4230729443738787e-05}, {\"label\": \"neutral\", \"score\": 0.010093500042319301}]"
#       ]
#     }
#   ]
# }
```

### NLP (Natural Language Processing)

URL: `/v2/models/nlp/versions/1/infer`

Input Names:

- `data`: Text data

Output Names:

- `results`: JSON dumped predictions with the `label` and `score` for each emotion

```python
# ...
text_data = "I am very happy today!"
request_data = json.dumps(
    {
        "id": "1",
        "inputs": [
            {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [text_data]},
        ],
    }
)
url = "http://localhost:8000/v2/models/nlp/versions/1/infer"
# ...
# Example output:
# {
#   "id": "193",
#   "model_name": "nlp",
#   "model_version": "1",
#   "outputs": [
#     {
#       "name": "results",
#       "datatype": "BYTES",
#       "shape": [],
#       "data": [
#         "[{\"label\": \"joy\", \"score\": 0.8967722058296204}, {\"label\": \"excitement\", \"score\": 0.037859223783016205}, {\"label\": \"admiration\", \"score\": 0.02824225090444088}, {\"label\": \"neutral\", \"score\": 0.027330709621310234}, {\"label\": \"gratitude\", \"score\": 0.02417466975748539}, {\"label\": \"relief\", \"score\": 0.022464321926236153}, {\"label\": \"approval\", \"score\": 0.021864689886569977}, {\"label\": \"love\", \"score\": 0.014073355123400688}, {\"label\": \"caring\", \"score\": 0.011970801278948784}, {\"label\": \"amusement\", \"score\": 0.009650727733969688}, {\"label\": \"optimism\", \"score\": 0.006543426308780909}, {\"label\": \"realization\", \"score\": 0.006097565405070782}, {\"label\": \"pride\", \"score\": 0.005770173855125904}, {\"label\": \"annoyance\", \"score\": 0.005515581928193569}, {\"label\": \"disapproval\", \"score\": 0.0043140980415046215}, {\"label\": \"confusion\", \"score\": 0.0036673692520707846}, {\"label\": \"sadness\", \"score\": 0.0036307028494775295}, {\"label\": \"anger\", \"score\": 0.003237350843846798}, {\"label\": \"desire\", \"score\": 0.0026491915341466665}, {\"label\": \"curiosity\", \"score\": 0.0022887929808348417}, {\"label\": \"surprise\", \"score\": 0.002227753633633256}, {\"label\": \"disappointment\", \"score\": 0.0017585484310984612}, {\"label\": \"nervousness\", \"score\": 0.0016981943044811487}, {\"label\": \"grief\", \"score\": 0.0011259763268753886}, {\"label\": \"remorse\", \"score\": 0.0009548053494654596}, {\"label\": \"fear\", \"score\": 0.0009162503411062062}, {\"label\": \"embarrassment\", \"score\": 0.0008813319727778435}, {\"label\": \"disgust\", \"score\": 0.0006478808936662972}]"
#       ]
#     }
#   ]
# }
```
