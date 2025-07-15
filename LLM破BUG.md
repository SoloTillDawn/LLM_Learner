1 module transformers has no attribute LlamaForCausalLM

```
transformers==4.45.2      tokenizers==0.20.1
```

2 TypeError: __init__() got an unexpected keyword argument 'enable_lora'

```
transformers==4.31.0
peft==0.3.0
sentence_transformers==2.2.1
torch==2.0.1
sentencepiece==0.2.0
bitsandbytes-windows==0.37.5
cpm_kernels==1.0.11
datasets==2.14.6
```

3 raise NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.")
NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.

```
datasets==2.14.6
```

4 安装xFormers时遇到的问题

```python
# torch==2.0.1 <==> xformers==0.0.20
# 解决：https://blog.csdn.net/BigerBang/article/details/139685883
pip install  xformers==0.0.20 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

5 Error caught was: No module named 'triton'

windows

```python
# https://github.com/PrashantSaikia/Triton-for-Windows目前只支持python3.10
# https://www.cnblogs.com/xing9/p/18394850
pip install cmake
pip install triton
```

6 resolve_trust_remote_code signal.signal(signal.SIGALRM, _raise_timeout_error) AttributeError: module 'signal' has no attribute 'SIGALRM'

没写上trust_remote_code=True导致的

```
model = transformers.AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=r'C:\Users\16776\Desktop\STD_Deep_Learning\HuggingfaceModel\chatglm2v1-6b',
    # trust_remote_code=True
)
```

7.大模型推理时提示cutlassF: no kernel found to launch!

使用chatglm4-6b、qianwen2.5-7b的模型进行推理，推理错误。增加如下torch的相关设置即可。

```
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
```

8. deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.7 does not match the version torch was compiled with 12.1, unable to compile cuda/cpp extensions without a matching cuda version.

   ```python
   修改文件
   /home/aha/miniconda3/envs/zhangzc/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py

   if sys_cuda_version != torch_cuda_version:
           return True   ### 添加一行这个
           if (cuda_major in cuda_minor_mismatch_ok and sys_cuda_version in cuda_minor_mismatch_ok[cuda_major]
                   and torch_cuda_version in cuda_minor_mismatch_ok[cuda_major]):
               print(f"Installed CUDA version {sys_cuda_version} does not match the "
                     f"version torch was compiled with {torch.version.cuda} "
                     "but since the APIs are compatible, accepting this combination")
               return True
   ```

9. [rank0]: ValueError: You can't train a model that has been loaded with `device_map='auto'` in any distributed mode. Please rerun your script specifying `--num_processes=1` or by launching with `python {{myscript.py}}`.

   ```
   device_map='auto'会自动将模型的不同部分分配到可用的GPU上。torchrun会自动管理设备分配，与device_map='auto'冲突。
   ```

   ​

10. ？

11. ？

12. ？