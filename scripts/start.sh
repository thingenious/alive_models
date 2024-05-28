#!/usr/bin/env sh

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
LD_LIBRARY_PATH="$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
python3 -m app
