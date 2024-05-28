"""Example usage/client for the ASR model."""

import argparse
import base64
import json
import os
import secrets
import tempfile
import threading

import av
import httpx
from playsound import playsound


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the ASR model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=str, help="Audio file to transcribe")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


def to_mono_16k_pcm(audio_file: str) -> bytes:
    """Convert the audio file to mono 16k PCM."""
    with tempfile.NamedTemporaryFile() as temp_file:
        with av.open(audio_file) as in_container:
            in_stream = in_container.streams.audio[0]
            with av.open(temp_file.name, "w", format="wav") as out_container:
                out_stream = out_container.add_stream(
                    "pcm_s16le",
                    rate=16000,
                    layout="mono",  # type:ignore
                )
                for frame in in_container.decode(in_stream):
                    for packet in out_stream.encode(frame):  # type:ignore
                        out_container.mux(packet)
        return temp_file.read()


def get_prediction(url: str, b64_data: str) -> None:
    """Get the prediction from the server."""
    headers = {"Content-Type": "application/json"}
    initial_prompt = ""
    inputs = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [b64_data]},
        {"name": "initial_prompt", "shape": [1, 1], "datatype": "BYTES", "data": [initial_prompt]},
    ]
    request_data = json.dumps(
        {
            "id": str(secrets.randbits(8)),
            "inputs": inputs,
        }
    )
    client = httpx.Client()
    # pylint: disable=broad-except,too-many-try-statements
    try:
        response = client.post(url, headers=headers, content=request_data, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        prediction_dicts = []
        outputs = response_data["outputs"]
        print(json.dumps(response_data))
        for entry in outputs:
            if "name" in entry and entry["name"] == "text":
                prediction_dicts.append(entry)
            elif "name" in entry and entry["name"] == "segments":
                # the data is a json dict, parse it (just for for better display when we json.dump it)
                _entry = dict(entry)
                segments = json.loads(entry["data"][0])
                _entry["data"][0] = segments
                prediction_dicts.append(_entry)
        print(json.dumps(prediction_dicts))
    except BaseException as error:
        print(f"Error sending request: {error}")


def main() -> None:
    """Parse the command line arguments and get the prediction."""
    args = cli().parse_args()
    if not os.path.exists(args.audio):
        print(f"File not found: {args.audio}")
        return
    _port = f":{args.port}" if args.port not in [80, 443] else ""
    model_name = "asr"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    try:
        audio_bytes = to_mono_16k_pcm(args.audio)
    except BaseException as error:  # pylint: disable=broad-except
        print(f"Error converting audio file: {error}")
        return
    try:
        data_str = base64.b64encode(audio_bytes).decode("utf-8")
    except BaseException as error:  # pylint: disable=broad-except
        print(f"Error reading audio file: {error}")
        return
    preview_thread = threading.Thread(target=playsound, args=(args.audio,))
    preview_thread.start()
    get_prediction(url, data_str)
    if preview_thread.is_alive():
        preview_thread.join()


if __name__ == "__main__":
    main()
