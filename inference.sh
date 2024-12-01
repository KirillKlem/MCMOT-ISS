#!/usr/bin/env bash

pip install -r requirements.txt
PYTHONPATH=./ python inference.py --videos_dir /private_test/ -m assets -s results