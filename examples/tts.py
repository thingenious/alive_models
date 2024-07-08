"""Example usage/client for the NLP model."""

import argparse
import base64
import json
import os
import secrets
import shutil
import wave
from datetime import datetime
from typing import Any, Dict

import httpx
from playsound import playsound


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the NLP model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    parser.add_argument("--speaker-index", type=int, help="Speaker index", default=None)
    parser.add_argument("--speaker-description", type=str, help="Speaker description", default=None)
    parser.add_argument("--speaker-name", type=str, help="Speaker name", default=None)
    parser.add_argument("--speaker-gender", type=str, help="Speaker gender", default=None)
    parser.add_argument("--speaker-language", type=str, help="Speaker language", default=None)
    parser.add_argument("--speaker-locale", type=str, help="Speaker locale", default=None)
    parser.add_argument("--speaker-pitch", type=str, help="Speaker pitch. (Default: +0Hz)", default=None)
    parser.add_argument("--speaker-rate", type=str, help="Speaker rate. (Default: +0%)", default=None)
    parser.add_argument("--speaker-volume", type=str, help="Speaker volume (Default: +0%)", default=None)
    parser.add_argument("--output-dir", type=str, help="Directory to store the speech", default=None)
    return parser


# pylint: disable=too-many-locals
def get_speech(url: str, text: str, output_dir: str | None = None, **kwargs: Any) -> None:
    """Get the speech from the server."""
    with httpx.Client() as client:
        headers = {"Content-Type": "application/json"}
        request_id = secrets.randbits(8)
        inputs = [
            {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [text]},
        ]
        request_data_dict: Dict[str, Any] = {
            "id": str(request_id),
            "inputs": inputs,
        }
        parameters = {}
        if kwargs:
            parameters = {key: value for key, value in kwargs.items() if value}
        if parameters:
            request_data_dict["parameters"] = parameters
        request_data = json.dumps(request_data_dict)
        # pylint: disable=broad-except,too-many-try-statements
        try:
            response = client.post(url, headers=headers, content=request_data, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            output_data = response_data["outputs"][0]["data"][0]
            audio = base64.b64decode(output_data)
            wav_out = wave.open("speech.wav", "wb")
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(16000)
            wav_out.writeframes(audio)
            wav_out.close()
        except BaseException as exc:
            print(f"Error: {exc}")
        else:
            if output_dir is not None and os.path.exists(output_dir):
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
                dest = os.path.join(output_dir, filename)
                shutil.copy("speech.wav", dest)
            playsound("speech.wav")
        finally:
            if os.path.exists("speech.wav"):
                os.remove("speech.wav")


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    model_name = "tts"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    text = args.text
    output_dir = args.output_dir
    kwargs = {
        "speaker_index": args.speaker_index,
        "speaker_description": args.speaker_description,
        "speaker_name": args.speaker_name,
        "speaker_gender": args.speaker_gender,
        "language": args.speaker_language,
        "locale": args.speaker_locale,
        "pitch": args.speaker_pitch,
        "rate": args.speaker_rate,
        "volume": args.speaker_volume,
    }
    get_speech(url, text, output_dir, **kwargs)


if __name__ == "__main__":
    main()
