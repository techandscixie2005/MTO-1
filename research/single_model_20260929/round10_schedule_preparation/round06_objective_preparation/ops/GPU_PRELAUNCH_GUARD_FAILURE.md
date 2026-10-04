# Preserved prelaunch provenance failure

The first invocation of `run_gpu_preflight.py --gpu 1` on 2026-10-01 exited 1 before admission, stage-directory creation, child launch, TRAIN decoding or optimizer updates. The exact source-review gate attempted to hash the reviewed Round06 `ENVIRONMENT.json`, which had not yet been mirrored from local storage.

Command (server working directory `round06_objective_preparation`):

```sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 /usr/bin/python3 run_gpu_preflight.py --gpu 1
```

Observed error:

```text
File "run_gpu_preflight.py", line 14, in main
    for path,digest in review['source_hashes'].items():assert sha(path)==digest
File "common.py", line 14, in sha
    with Path(path).open('rb') as stream:
FileNotFoundError: [Errno 2] No such file or directory: '/home/inspur/MTO-1/research/single_model_20260929/round06_objective_preparation/ENVIRONMENT.json'
```

History independently mirrored the already-reviewed local file, SHA256 `57c2f536b2af90baf8816a0924be5a7511e88ba0f35fbeda5b55e95fdf2540a9`, and rechecked all 116 technical source bindings. The bound review remains `fbf76a414d89a459cf5fa3c956544a698b1f564c856622c107fa99b6dc592605`. No source, settings, numerical threshold or update budget changed. The explicit next invocation is the first actual admitted fixture; this failed gate created no model state.
