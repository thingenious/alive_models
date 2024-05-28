"""Example usage/client for the NLP model."""

import argparse
import json
import secrets

import httpx


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the NLP model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=str, help="Text to analyze")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


def get_prediction(url: str, text: str) -> None:
    """Get the prediction from the server."""
    headers = {"Content-Type": "application/json"}
    request_id = secrets.randbits(8)
    inputs = [
        {"name": "text", "shape": [1, 1], "datatype": "BYTES", "data": [text]},
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
        print(json.dumps(response_data))
    except BaseException as exc:
        print(f"Error: {exc}")
        if response is not None:
            print(response.text)
    finally:
        client.close()


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    model_name = "nlp"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    get_prediction(url, args.text)


if __name__ == "__main__":
    main()
