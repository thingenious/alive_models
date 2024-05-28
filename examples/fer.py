"""Example usage/client for the FER model."""

import argparse
import base64
import json
import secrets
from io import BytesIO

import httpx
from PIL import Image


def cli() -> argparse.ArgumentParser:
    """Get the command line interface for the FER model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=argparse.FileType("rb"), help="Image file to predict")
    parser.add_argument("--host", default="localhost", help="Host of the server")
    parser.add_argument("--port", default=8000, type=int, help="Port of the server")
    parser.add_argument("--scheme", default="http", help="Scheme of the server")
    return parser


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
        response = client.post(url, headers=headers, content=request_data, timeout=30)
        response.raise_for_status()
        response_data = response.json()
        print(json.dumps(response_data))
    except BaseException as exc:
        print(f"Error: {exc}")
    finally:
        client.close()


def main() -> None:
    """Run the main function."""
    args = cli().parse_args()
    _port = f":{args.port}" if args.port not in {80, 443} else ""
    model_name = "fer"
    model_version = 1
    url = f"{args.scheme}://{args.host}{_port}/v2/models/{model_name}/versions/{model_version}/infer"
    with args.image as image_file:
        try:
            image = Image.open(image_file)
        except BaseException as exc:  # pylint: disable=broad-except
            print(f"Error: {exc}")
            return
        image.show()
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        get_prediction(url, img_str)


if __name__ == "__main__":
    main()
