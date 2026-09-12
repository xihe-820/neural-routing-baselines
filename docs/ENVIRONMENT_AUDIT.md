# Environment audit

服务器状态：`NOT_RUN`。以下仅为 WSL 本地已有 gan 环境，不代表 cp311_base；本轮没有安装、更改任何依赖。

Environment label: `local-gan-not-cp311_base`.
This report describes only the interpreter and host recorded below. Imports do not prove compiled CUDA operations or method compatibility.

- Timestamp: 2026-09-11T08:46:39.452238+00:00
- Host: xihe
- Python: 3.10.20
- Executable: `/home/xihe/miniconda3/envs/gan/bin/python`
- Platform: Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.43

| Package | Installed | Version | Import | Error |
|---|---|---|---|---|
| torch | True | 2.5.1 | True |  |
| numpy | True | 1.24.3 | True |  |
| scipy | False | None | False | ModuleNotFoundError: No module named 'scipy' |
| networkx | True | 3.4.2 | True |  |
| torch-geometric | False | None | False | ModuleNotFoundError: No module named 'torch_geometric' |
| torch-scatter | False | None | False | ModuleNotFoundError: No module named 'torch_scatter' |
| torch-sparse | False | None | False | ModuleNotFoundError: No module named 'torch_sparse' |
| torch-cluster | False | None | False | ModuleNotFoundError: No module named 'torch_cluster' |
| torch-spline-conv | False | None | False | ModuleNotFoundError: No module named 'torch_spline_conv' |
| tensordict | False | None | False | ModuleNotFoundError: No module named 'tensordict' |
| torchrl | False | None | False | ModuleNotFoundError: No module named 'torchrl' |
| rl4co | False | None | False | ModuleNotFoundError: No module named 'rl4co' |
| lightning | False | None | False | ModuleNotFoundError: No module named 'lightning' |
| pytorch-lightning | False | None | False | ModuleNotFoundError: No module named 'pytorch_lightning' |
| random-insertion | False | None | False | ModuleNotFoundError: No module named 'random_insertion' |
| protobuf | False | None | False | ModuleNotFoundError: No module named 'google' |
| hydra-core | False | None | False | ModuleNotFoundError: No module named 'hydra' |
| omegaconf | False | None | False | ModuleNotFoundError: No module named 'omegaconf' |
| huggingface-hub | True | 1.17.0 | True |  |
| pyvrp | False | None | False | ModuleNotFoundError: No module named 'pyvrp' |
| vrplib | False | None | False | ModuleNotFoundError: No module named 'vrplib' |
| ml4co-kit | False | None | False | ModuleNotFoundError: No module named 'ml4co_kit' |
| tensorboard-logger | False | None | False | ModuleNotFoundError: No module named 'tensorboard_logger' |

受限sandbox中的CUDA discovery（只反映该执行上下文）：
```json
{
  "torch_version_cuda": "12.1",
  "available": false,
  "gpu_names": []
}
```

pip check:
```json
{
  "returncode": 0,
  "stdout": "No broken requirements found.\n",
  "stderr": "WARNING: The directory '/home/xihe/.cache/pip' or its parent directory is not owned or is not writable by the current user. The cache has been disabled. Check the permissions and owner of that directory. If executing pip with sudo, you should use sudo's -H flag.\n"
}
```


## WSL host GPU：LOCAL_VERIFIED

2026-09-12 01:23:34（Asia/Shanghai）在获准的只读host检查中，`nvidia-smi`成功：NVIDIA GeForce RTX 4060 Laptop GPU、8188MiB、driver591.86；gan Torch2.5.1的`torch.cuda.is_available()`为True并返回同一GPU。之前sandbox报NVML access blocked并返回False，不能据此把WSL判成CPU-only。证据保存在 `artifacts/environment/local_gpu.json` 和 [environment manifest](../manifests/environment.yaml)。nvidia-smi的CUDA13.1是驱动支持上限；gan Torch build为CUDA12.1。尚未执行模型GPU forward。

## Existing reference-test environment

使用已有 `/home/xihe/miniconda3/envs/ml4co_venv/bin/python`：Python3.10.20、Torch2.12.0+cpu、NumPy2.2.6、SciPy1.15.3、ML4CO-Kit0.5.4。该解释器承担数据解析与reference tests；不把它的依赖版本混入gan strict-load记录。已有Kit checkout为d22fb8be071156a15836d5ed813123c250800fa8，预先dirty的Concorde setup.py未修改；routing task源码SHA单独记录在数据manifest。

以上都不是cp311_base/RTX4090当前事实。gan pip check返回0仅说明已安装包的声明依赖未破损，不能证明七个baseline依赖齐全。本轮无任何package install、核心stack升级/降级或新建legacy环境。
