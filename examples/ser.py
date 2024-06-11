"""Example usage/client for the SER model."""

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
    """Get the command line interface for the SER model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=str, help="Audio file to analyze")
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
    request_id = secrets.randbits(8)
    inputs = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [b64_data]},
    ]
    request_data = json.dumps(
        {
            "id": str(request_id),
            "inputs": inputs,
        }
    )
    client = httpx.Client()
    # pylint: disable=broad-except,too-many-try-statements
    try:
        response = client.post(url, headers=headers, content=request_data, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        response_dicts = json.loads(response_data["outputs"][0]["data"][0])
        print(json.dumps(response_dicts, indent=2))
        most_probable = max(response_dicts, key=lambda item: item["score"])
        print(f"Most probable sentiment: {most_probable}")
    except BaseException as exc:
        print(f"Error: {exc}")
    finally:
        client.close()


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    if not os.path.exists(args.audio):
        print(f"Error: {args.audio} does not exist")
        return
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    model_name = "ser"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    # pylint: disable=broad-except,too-many-try-statements
    try:
        audio = to_mono_16k_pcm(args.audio)
        b64_data = base64.b64encode(audio).decode("utf-8")
    except BaseException as exc:
        print(f"Error: {exc}")
        return
    preview_thread = threading.Thread(target=playsound, args=(args.audio,))
    preview_thread.start()
    get_prediction(url, b64_data)


if __name__ == "__main__":
    main()
