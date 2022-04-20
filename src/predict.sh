#!/usr/bin/env sh
set -e
set -v
python src/runner.py test --work_dir work --test_data $1 --test_output $2
