# Models

By default all models are enabled: ASR, FER, NLP, SER. You can disable one or more by setting the `ALIVE_MODELS` environment variable to a comma-separated list of the models you want to enable.

Default: `ALIVE_MODELS=asr,fer,nlp,ser`

## Automatic Speech Recognition (ASR)

Model: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)  
Default Model Size: [`distil-large-v3`](https://huggingface.co/distil-whisper/distil-large-v3)  
Override it with `ALIVE_MODELS_ASR_MODEL_SIZE` environment variable.

::: app.models.asr
    options:
        show_root_toc_entry: false
        members:
            - ASR_INPUTS
            - ASR_OUTPUTS
            - asr_infer_fn
            - get_transcription

## Facial Emotion Recognition (FER)

Library: [serengil/deepface](https://github.com/serengil/deepface)  
Default detector backend: [`yolov8`](https://github.com/ultralytics/ultralytics)  
Override it with `ALIVE_MODELS_FER_MODEL_DETECTOR_BACKEND` environment variable.

::: app.models.fer
    options:
        include: FER_MODEL_INPUTS
        show_root_toc_entry: false
        members:
            - FER_INPUTS
            - FER_OUTPUTS
            - fer_infer_fn
            - get_image_analysis

## Natural Language Processing (NLP)

Library: [huggingface/transformers](https://github.com/huggingface/transformers)  
Default model: [`SamLowe/roberta-base-go_emotions-onnx`]([SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)) file: `onnx/model_quantized.onnx`  
Override it with `ALIVE_MODELS_NLP_MODEL_REPO` and/or `ALIVE_MODELS_NLP_MODEL_FILE` environment variables.

::: app.models.nlp
    options:
        show_root_toc_entry: false
        members:
            - NLP_INPUTS
            - NLP_OUTPUTS
            - nlp_infer_fn
            - get_text_sentiment

## Speech Emotion Recognition (SER)

::: app.models.ser
    options:
        show_root_toc_entry: false
        members:
            - SER_INPUTS
            - SER_OUTPUTS
            - ser_infer_fn
            - get_audio_analysis
