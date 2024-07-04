"""Example usage/client for the LID model."""

import argparse
import json
import secrets
from typing import Dict

import httpx


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the LID model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


def print_most_probable(most_probable_dict: Dict[str, str | float]) -> None:
    """Print the most probable language."""
    most_probable = str(most_probable_dict.get("label", "unknown")).split("_")[0]
    if most_probable in ("en", "eng"):
        print("Most probable language: English")
    elif most_probable in ("es", "spa"):
        print("Most probable language: Spanish")
    elif most_probable in ("fr", "fra"):
        print("Most probable language: French")
    elif most_probable in ("de", "deu"):
        print("Most probable language: German")
    elif most_probable in ("it", "ita"):
        print("Most probable language: Italian")
    elif most_probable in ("ru", "rus"):
        print("Most probable language: Russian")
    elif most_probable in ("el", "gr", "ell"):
        print("Most probable language: Greek")
    elif most_probable in ("tr", "tur"):
        print("Most probable language: Turkish")
    else:
        print(f"Most probable language: {most_probable}")


def get_prediction(url: str, text: str) -> None:
    """Get the prediction from the server."""
    headers = {"Content-Type": "application/json"}
    request_id = secrets.randbits(8)
    inputs = [
        {"name": "data", "shape": [1, 1], "datatype": "BYTES", "data": [text]},
    ]
    request_data = json.dumps(
        {
            "id": str(request_id),
            "inputs": inputs,
        }
    )
    client = httpx.Client()
    # pylint: disable=broad-except,too-many-try-statements
    response = None
    try:
        response = client.post(url, headers=headers, content=request_data, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        response_dicts = json.loads(response_data["outputs"][0]["data"][0])
        print(json.dumps(response_dicts, indent=2))
        if not isinstance(response_dicts, list):
            response_dicts = [response_dicts]
        most_probable_dict = max(response_dicts, key=lambda item: item["score"])
        print_most_probable(most_probable_dict)
    except BaseException as exc:
        print(f"Error: {exc}")
        if response is not None:
            print(response.content)
    finally:
        client.close()


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    model_name = "lid"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    get_prediction(url, args.text)


if __name__ == "__main__":
    main()
