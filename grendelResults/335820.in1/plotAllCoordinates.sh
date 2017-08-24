#!/bin/bash

DIR="$(pwd)"

for file in ${DIR}/opt*
do
    python3 ../../plotCoordinates.py ${file}
done

