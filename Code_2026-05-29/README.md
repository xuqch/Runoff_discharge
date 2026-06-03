# Code_2026-05-22

这个目录是当前使用的 PhaseH / HydroAI 实验代码仓库，包含数据预处理、H5 构建、训练、验证和逐格点推理脚本。

## 目录概览

### 训练与验证入口

- `train_global.py`
  - 主训练入口。
  - 负责读取配置、准备 H5 数据、构建 DataLoader、初始化模型、执行单卡或 DDP 训练、保存 checkpoint。

- `validation.py`
  - 主验证入口。
  - 使用训练好的 checkpoint 和评估期 H5 数据，输出 basin 级 CSV、NetCDF 和评估指标汇总。

### 配置与公共函数

- `config.py`
  - 命令行参数定义和配置整理入口。
  - 负责把 `--data_dir`、`--q_file`、`--run_dir` 等参数整理成训练、验证、推理统一使用的配置字典。

- `common.py`
  - 训练和验证共用工具函数。
  - 包括 loss 构建、checkpoint 查找、输出目录创建、时间格式转换、metrics 计算、CUDA cache 清理等通用逻辑。

- `utils.py`
  - 底层工具库。
  - 包括 scaler 读写、NetCDF 数据展开、静态/动态变量读取、PET 估算、面积向量构建、随机种子设置等基础函数。

### 数据预处理与 H5 构建

- `fit_scalers.py`
  - 扫描 basin nc 文件，拟合动态变量、静态变量和 qobs 的 scaler。
  - qobs 直接从 basin forcing nc 中读取，不再依赖单独 qobs 文件。

- `build_h5.py`
  - 把原始 basin nc 数据转换成训练和验证使用的 per-basin H5。
  - 负责切训练时段，生成 `target_data / target_valid / target_dates / target_idx / q_stds` 等数据集，并写出 `manifest.csv`。

- `dataset_global.py`
  - H5 数据集定义。
  - 当前核心类是 `PerBasinBlockH5Dataset`，按 basin 和 target block 读取样本，支持 block 级高效切片，不压缩 NaN qobs 时间轴。

- `balanced_block_sampler.py`
  - DDP 训练时使用的 block sampler。
  - 用于多卡下更均衡地分配 basin block，减少每卡负载差异。

### 模型与损失

- `model_hydroai_basin.py`
  - 当前 basin/block 训练主模型。
  - 负责生成 runoff、执行 routing，并适配 basin block 训练所需的 `basin_meta` 输入。

- `model.py`
  - 较早的模型定义，主要被逐格点 inference 脚本复用。
  - 当前 basin/block 训练主入口不是走这个文件，而是走 `model_hydroai_basin.py`。

- `loss.py`
  - 损失函数定义。
  - 包括 NSE、HydroLoss、HydroMFMLoss、PeakFlowLoss 等，当前统一基于原始 `q_true + q_valid mask` 计算，不再依赖标准化观测目标链。

### 推理相关

- `inference_unit.py`
  - 推理公共模块。
  - 抽取了 single/multi inference 共享逻辑，例如模型加载、年度切片、静态变量读取、valid grid 构建、逐格点或小批量格点预测。

- `inference_global.py`
  - 单 GPU 年尺度逐格点推理入口。
  - 逐年读取数据，逐格点或按 `grid_batch_size` 小批量生成空间 runoff 结果。

- `inference_global_parallel.py`
  - 多 GPU / 多进程年尺度逐格点推理入口。
  - 把有效格点切 shard 后分给不同 GPU worker，每个 worker 写临时 part 文件，最后主进程合并成年度 NetCDF。

### 运行脚本

- `run.sh`
  - 常用命令示例集合。
  - 包含训练、验证、推理的示例调用，适合作为手工运行时的参考。

- `launch_ddp_train_phaseh.sh`
  - 极简 DDP 启动器。
  - 负责设置 `CUDA_VISIBLE_DEVICES`、`NPROC_PER_NODE`，然后用 `torchrun` 启动 `train_global.py`。

### 可视化与结果分析

- `Plot_val_result.py`
  - 验证结果绘图脚本。
  - 用于读取验证输出并生成图表或可视化结果。

### 数据目录

- `data/`
  - 仓库内随代码存放的辅助数据目录。
  - 当前已知包含 `basins_for_train.csv` 这类 basin 列表文件。

### 其他

- `__pycache__/`
  - Python 编译缓存目录，可忽略。

## 当前主流程建议

### 1. 数据准备

1. 在 `data_dir/Basins_data/` 下准备每个 basin 的 nc 文件。
2. 确保 nc 文件内同时包含：
   - 动态 forcing 变量
   - 静态变量
   - `qobs_var` 对应的流量观测变量
3. 准备 basin 列表文件，例如 `data/basins_for_train.csv`。

### 2. 训练

1. `train_global.py` 会先调用 `fit_scalers.py` 拟合 scaler。
2. 然后调用 `build_h5.py` 生成 per-basin H5。
3. 再通过 `dataset_global.py` + `balanced_block_sampler.py` 组织 block 训练。
4. 模型主体在 `model_hydroai_basin.py`，损失在 `loss.py`。

### 3. 验证

1. `validation.py` 会按评估时间段重新构建 eval H5。
2. 使用训练好的 checkpoint 逐 basin 输出：
   - `*.csv`
   - `*.nc`
   - `metrics_summary.csv`
   - `eval_summary.json`

### 4. 推理

1. 单 GPU 推理用：
   - `inference_global.py`
2. 多 GPU 推理用：
   - `inference_global_parallel.py`
3. 共享逻辑统一放在：
   - `inference_unit.py`

## Inference 使用建议

### 怎么选

1. 如果你只有 1 张 GPU，或者先想确认模型、输入数据、输出 NetCDF 是否正常，优先用 `inference_global.py`。
2. 如果你有多张 GPU，且年度推理耗时较长，优先用 `inference_global_parallel.py`。
3. 如果你正在排查 NaN、坐标、年份切片、输出变量名之类的问题，先用单卡脚本更容易定位。
4. 如果你已经确认单卡结果正常，再切到并行脚本做正式整年批量推理更稳妥。

### 单卡推荐参数

- 首次调试建议：
  - `--infer_start_year 1999 --infer_end_year 1999`
  - `--grid_batch_size 1`
- 单卡稳定后可尝试：
  - `--grid_batch_size 64`
  - `--grid_batch_size 128`
  - `--grid_batch_size 256`
- 如果显存不足或出现 OOM：
  - 把 `--grid_batch_size` 降回 `1 / 32 / 64`

### 并行推荐参数

- 首次并行测试建议：
  - `--cuda_devices 0,1`
  - `--infer_start_year 1999 --infer_end_year 1999`
  - `--grid_batch_size 1`
- 多卡稳定后可尝试：
  - `--cuda_devices 0,1,2,3`
  - `--grid_batch_size 64`
  - `--grid_batch_size 128`
- 如果某个 worker 显存溢出：
  - 优先降低 `--grid_batch_size`
  - 一般不需要先改 `cuda_devices` 分片逻辑

### 额外说明

- `grid_batch_size=1` 时，行为最接近旧版逐格点推理，适合保守验证。
- `grid_batch_size>1` 时，GPU 利用率通常更高，但更容易吃显存。
- 并行脚本是“多个 GPU 分不同格点 shard”，不是多个 worker 同时写同一个 NetCDF。
- 如果想保留中间分片结果排查问题，可以加：
  - `--keep_inference_parts`
- 如果临时分片文件想放到单独目录，可以加：
  - `--inference_tmp_dir <path>`

## 备注

- 当前仓库已经移除了旧的标准化观测目标数据链。
- qobs 缺失值通过 `q_valid` mask 参与 loss 和评估，不在 Dataset 层压缩时间轴。
- target 数据读取已经按 block 切片优化，不再为每个 block 反复读取整 basin 的完整 target 数组。
