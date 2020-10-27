# Qianliyan (千里眼) Solution to the SpaceNet 7 Challenge

# Download Data and Create Environment

1. Download SpaceNet 7 Data
    - `cd /root/workspace/data`
    - `aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .`
    - `aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .`
    - `tar -xvf SN7_buildings_train.tar.gz`
    - `tar -xvf SN7_buildings_test_public.tar.gz`
2. Build and launch the docker container, which relies upon [Solaris](https://solaris.readthedocs.io/en/latest/) (a GPU-enabled machine and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are recommended):
    - `nvidia-docker build -t qianliyan docker`
    - `NV_GPU=0 nvidia-docker run -it -v /local_data:/local_data  -ti --ipc=host --name sn7_gpu0 sn7_baseline_image`
    - `conda activate solaris`

# Train Stage

```bash
./train.sh
```

# Test Stage

```bash
./test.sh
```
