#!/usr/bin/env sh

_HERE="$(dirname "$(readlink -f "$0")")"
_PARENT="$(dirname "$_HERE")"
_PROJECT_ROOT="${_PARENT}"

if [ ! -d "${_PARENT}/app" ] && [ -d "$_HERE/app" ]; then
    # in container?
    _PROJECT_ROOT="${_HERE}"
fi

cd "${_PROJECT_ROOT}" || exit 1

# https://github.com/SYSTRAN/faster-whisper/issues/516#issuecomment-2048492428
LD_LIBRARY_PATH="$(python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
echo "Initializing the app environment..."
# a simple import to make sure the files are downloaded
python3 -c "from app import models"
