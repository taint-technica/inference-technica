docker run --gpus all --name xinference \
    -p 9997:9997 --entrypoint tail \
    -d \
    -v "$(pwd)":/opt/inference xinference:latest \
    -f /dev/null