"""Example usage/client for all the available models."""

import argparse
import base64
import json
import os
import platform
import secrets
import subprocess  # nosemgrep # nosec
import tempfile
import threading
from io import BytesIO
from typing import Any, Dict, List

import av
import httpx
from PIL import Image
from playsound import playsound


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the ASR model.

    Returns
    -------
    argparse.ArgumentParser
        The CLI parser.

    Notes
    -----
    The following arguments are used:
    - `video`: Path to the video file to analyze.
    - `--host`: Host of the server.
    - `--port`: Port of the server.
    - `--scheme`: Scheme of the server.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=argparse.FileType("rb"), help="Video file to analyze")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


def to_mono_16k_pcm(video_file: str) -> bytes:
    """Convert the video file to mono 16k PCM.

    Parameters
    ----------
    video_file : str
        Path to the video file.

    Returns
    -------
    bytes
        The audio data.
    """
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
    """Get the image snapshot from the video file.

    Parameters
    ----------
    video_file : str
        Path to the video file.

    Returns
    -------
    Image
        The image snapshot.
    """
    with av.open(video_file) as container:
        for frame in container.decode(video=0):
            return frame.to_image()  # type:ignore


def make_request(
    base_url: str,
    model_name: str,
    model_version: int,
    input_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Make a request to the server.

    Parameters
    ----------
    base_url : str
        The base URL of the server.
    model_name : str
        The model name.
    model_version : int
        The model version.
    input_data : List[Dict[str, Any]]
        The input data.

    Returns
    -------
    Dict[str, Any]
        The response data.
    """
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
    """Get the ASR prediction from the server.

    Parameters
    ----------
    base_url : str
        The base URL of the server.
    audio_data : str
        The audio data (base64 encoded).

    Returns
    -------
    str
        The transcript.
    """
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [audio_data]},
    ]
    response_data = make_request(base_url, "asr", 1, input_data)
    response_dicts = json.loads(response_data["outputs"][0]["data"][0])
    transcript = ""
    for entry in response_dicts:
        text = entry.get("text", "")
        if text:
            transcript += text
    print(f"ASR prediction:\n{json.dumps(response_dicts)}\n")
    print(f"Transcript: \n{transcript}\n")
    return transcript


def get_ser_prediction(base_url: str, audio_data: str) -> Dict[str, Any]:
    """Get the SER prediction from the server.

    Parameters
    ----------
    base_url : str
        The base URL of the server.
    audio_data : str
        The audio data (base64 encoded).

    Returns
    -------
    Dict[str, Any]
        The SER prediction.
    """
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [audio_data]},
    ]
    response_data = make_request(base_url, "ser", 1, input_data)
    return response_data


def get_fer_prediction(base_url: str, image: Image) -> Dict[str, Any]:
    """Get the FER prediction from the server.

    Parameters
    ----------
    base_url : str
        The base URL of the server.
    image : Image
        The image snapshot.

    Returns
    -------
    Dict[str, Any]
        The FER prediction.
    """
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_data = base64.b64encode(buff.getvalue()).decode("utf-8")
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [image_data]},
    ]
    response_data = make_request(base_url, "fer", 1, input_data)
    return response_data


def get_nlp_prediction(base_url: str, transcription: str) -> Dict[str, Any]:
    """Get the NLP prediction from the server.

    Parameters
    ----------
    base_url : str
        The base URL of the server.
    transcription : str
        The transcription.

    Returns
    -------
    Dict[str, Any]
        The NLP prediction.
    """
    input_data = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [transcription]},
    ]
    response_data = make_request(base_url, "nlp", 1, input_data)
    return response_data


def _open_native(video_file: str) -> None:
    """Open the video file on the default video player.

    Parameters
    ----------
    video_file : str
        Path to the video file.
    """
    if platform.system() == "Darwin":
        subprocess.call(("open", video_file))  # nosemgrep # nosec
    elif platform.system() == "Windows":
        # pylint: disable=no-member
        os.startfile(video_file)  # type:ignore # nosemgrep # nosec
    else:
        subprocess.call(("xdg-open", video_file), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_video(video_file: str) -> None:
    """Open the video file for preview.

    Parameters
    ----------
    video_file : str
        Path to the video file.
    """
    print("Opening video file for preview...")
    try:
        # playsound could also work with video files
        # it uses gstreamer's playbin on linux
        playsound(video_file)
    except BaseException:  # pylint: disable=broad-except
        _open_native(video_file)


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    preview_thread = threading.Thread(target=open_video, args=(args.video.name,))
    preview_thread.start()
    audio_data = base64.b64encode(to_mono_16k_pcm(args.video.name)).decode("utf-8")
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    base_url = f"{args.scheme}://{args.host}{_port}"
    transcript = get_asr_prediction(base_url, audio_data)
    image = get_video_image_snapshot(args.video.name)
    fer_prediction = get_fer_prediction(base_url, image)
    outputs: List[Dict[str, Any]] = json.loads(fer_prediction["outputs"][0]["data"][0])
    print(f"FER prediction:\n{json.dumps(fer_prediction)}\n")
    most_probable = max(outputs, key=lambda item: item["score"])
    print(f"Most probable FER sentiment: {most_probable}")
    nlp_prediction = get_nlp_prediction(base_url, transcript)
    outputs = json.loads(nlp_prediction["outputs"][0]["data"][0])
    print(f"NLP prediction:\n{json.dumps(nlp_prediction)}\n")
    most_probable = max(outputs, key=lambda item: item["score"])
    print(f"Most probable NLP sentiment: {most_probable}")
    ser_prediction = get_ser_prediction(base_url, audio_data)
    outputs = json.loads(ser_prediction["outputs"][0]["data"][0])
    most_probable = max(outputs, key=lambda item: item["score"])
    print(f"SER prediction:\n{json.dumps(ser_prediction)}\n")
    print(f"Most probable SER sentiment: {most_probable}")


if __name__ == "__main__":
    main()
