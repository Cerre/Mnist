

## Command to run when inside mnist/ folder
docker run -v $PWD:/app -it --rm mnist:v1 python src/mnist.py