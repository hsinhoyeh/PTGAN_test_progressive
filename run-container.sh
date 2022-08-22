#!/usr/bin/env bash

docker run -e NVIDIA_VISIBLE_DEVICES=0 \
    -itd -v /home/ubuntu/workspace:/workspace -v /home/ubuntu/models:/models pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# use docker exec <container-id> /bin/bash to access the container
