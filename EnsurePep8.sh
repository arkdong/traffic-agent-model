#!/bin/bash

if command -v black > /dev/null 2>&1; then
    echo "Black is installed"

else
    echo "Black is not installed, installing ..."
    pip install black

fi

python_files=($(find . -name "*.py"))

for file in "${python_files[@]}"; do
    black --line-length 79 ${file}
done