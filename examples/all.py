"""Example usage/client for all the available models."""

import argparse
import base64
import json
import os
import platform
import secrets
import subprocess  # nosemgrep # nosec
import tempfile
from io import BytesIO
from typing import Any, Dict, List

import av
import httpx
from PIL import Image


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the ASR model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=argparse.FileType("rb"), help="Video file to analyze")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


def to_mono_16k_pcm(video_file: str) -> bytes:
    """Convert the video file to mono 16k PCM."""
    with tempfile.NamedTemporaryFile() as temp_file:
        with av.open(video_file) as in_container:
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


def get_video_image_snapshot(video_file: str) -> Image:
    """Get the image snapshot from the video file."""
    with av.open(video_file) as container:
        for frame in container.decode(video=0):
            return frame.to_image()  # type:ignore


def make_request(
    base_url: str,
    model_name: str,
    model_version: int,
    input_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Make a request to the server."""
    headers = {"Content-Type": "application/json"}
    request_id = secrets.randbits(8)
    url = f"{base_url}/v2/models/{model_name}/versions/{model_version}/infer"
    request_data = json.dumps(
        {
            "id": str(request_id),
            "inputs": input_data,
        }
    )
    client = httpx.Client()
    # pylint: disable=broad-except,too-many-try-statements
    try:
        response = client.post(url, headers=headers, content=request_data, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        print(url)
        print(f"Error: {error}")
        return {"outputs": []}


def get_asr_prediction(base_url: str, audio_data: str) -> str:
    """Get the ASR prediction from the server."""
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [audio_data]},
        {"name": "previous_transcript", "shape": [1, 1], "datatype": "BYTES", "data": [""]},
    ]
    response_data = make_request(base_url, "asr", 1, input_data)
    prediction_dicts = []
    outputs = response_data["outputs"]
    transcription = None
    for entry in outputs:
        if "name" in entry and entry["name"] == "text":
            transcription = entry["data"][0]
            prediction_dicts.append(entry)
        elif "name" in entry and entry["name"] == "segments":
            _entry = dict(entry)
            # the data is a json dict, parse it (for better display)
            segments = json.loads(entry["data"][0])
            _entry["data"][0] = segments
            prediction_dicts.append(_entry)
    if not transcription:
        transcription = "No transcription available"
    prediction = {"outputs": prediction_dicts}
    print(f"ASR prediction:\n{json.dumps(prediction)}\n")
    return transcription


def get_ser_prediction(base_url: str, audio_data: str) -> None:
    """Get the SER prediction from the server."""
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [audio_data]},
    ]
    response_data = make_request(base_url, "ser", 1, input_data)
    print(f"SER prediction:\n{json.dumps(response_data)}\n")


def get_fer_prediction(base_url: str, image: Image) -> None:
    """Get the FER prediction from the server."""
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_data = base64.b64encode(buff.getvalue()).decode("utf-8")
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [image_data]},
    ]
    response_data = make_request(base_url, "fer", 1, input_data)
    print(f"FER prediction:\n{json.dumps(response_data)}\n")


def get_nlp_prediction(base_url: str, transcription: str) -> None:
    """Get the NLP prediction from the server."""
    input_data = [
        {"name": "text", "shape": [1, 1], "datatype": "BYTES", "data": [transcription]},
    ]
    response_data = make_request(base_url, "nlp", 1, input_data)
    print(f"NLP prediction:\n{json.dumps(response_data)}\n")


def open_video(video_file: str) -> None:
    """Open the video file on the default video player."""
    if platform.system() == "Darwin":
        subprocess.call(("open", video_file))  # nosemgrep # nosec
    elif platform.system() == "Windows":
        # pylint: disable=no-member
        os.startfile(video_file)  # type:ignore # nosemgrep # nosec
    else:
        subprocess.call(("xdg-open", video_file), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    print("Opening video file for preview...")
    open_video(args.video.name)
    audio_data = base64.b64encode(to_mono_16k_pcm(args.video.name)).decode("utf-8")
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    base_url = f"{args.scheme}://{args.host}{_port}"
    transcript = get_asr_prediction(base_url, audio_data)
    image = get_video_image_snapshot(args.video.name)
    get_fer_prediction(base_url, image)
    get_nlp_prediction(base_url, transcript)
    get_ser_prediction(base_url, audio_data)


if __name__ == "__main__":
    main()
