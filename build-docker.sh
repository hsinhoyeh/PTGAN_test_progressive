#!/usr/bin/env bash

TAG ?= latest
docker build -t footprintai/iccc-sp8:${TAG} -f Dockerfile .
docker push footprintai/iccc-sp8:${TAG}
