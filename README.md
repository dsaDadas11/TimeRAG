# TimeRAG Jupyter 教程（t/）

本目录包含一套按步骤执行的 Jupyter Notebook，用于完成 ERA5 温度数据下载、样本生成、向量化、可视化与 RAG 检索/数据集生成。

## 目录内容
- `1_tutorial_download.ipynb`：ERA5 数据下载（查询城市 1991–2023；目标城市汉中 2024）
- `2_tutorial_samples.ipynb`：滑动窗口样本生成（window=192, input=168, pred=24, step=6）
- `3_1_tutorial_vectors.ipynb`：样本向量化（Chronos + FAISS）
- `3_2_tutorial_visualize.ipynb`：可视化与数据验证
- `4_1_tutorial_rag.ipynb`：RAG 检索 + TopK 可视化
- `4_2_tutorial_rag_process.ipynb`：批量检索生成 RAG 数据集并划分 train/val/test
- `trian_xx.py`：预测模型，train_autoformer_fuse.py为融合预测模型，其余为baseline
## 运行前准备

### 1) Python 依赖（示例）
按需安装（CPU 版示例）：
```
pip install cdsapi xarray netCDF4 numpy pandas matplotlib tqdm faiss-cpu torch
```

另外需要安装并可导入 `ChronosPipeline` 的 `chronos` 库，并准备好本地模型目录。
ChronosPipeline安装：pip install git+https://github.com/amazon-science/chronos-forecasting.git
chronos模型下载：https://modelscope.cn/models/better464/times

### 2) CDS API 配置
下载 ERA5 前需配置 `.cdsapirc`。示例（请替换 key）：
```
url: https://cds.climate.copernicus.eu/api
key: <api-key>
```

### 3) 路径一致性（重要）
各 notebook 默认路径不完全一致，请确保它们指向同一数据根目录。

建议做法（任选其一）：
- **方案 A**：统一把所有 `DATA_ROOT`/`OUTPUT_DIR`/`SAMPLES_ROOT` 改成同一绝对路径（推荐）。
- **方案 B**：保证 notebook 工作目录与路径写法一致（注意 `TimeRAG/...` 相对路径）。

典型数据根目录建议为：
```
/storage/luoao/TimeRAG/shaanxi_cities_temperature
```

## 推荐执行顺序

1. **数据下载**
   - `1_tutorial_download.ipynb`
   - 输出：`<数据根目录>/<城市>_<年份>.npy` + `<城市>_<年份>_times.npy`
   - 查询城市：西安、郑州、太原、石家庄、济南、呼和浩特、银川、兰州、西宁、大同
   - 目标城市：汉中（2024）

2. **样本生成**
   - `2_tutorial_samples.ipynb`
   - window=192, input=168, pred=24, step=6
   - 输出目录：
     - `samples_w192_s6/query/`
     - `samples_w192_s6/target/`

3. **向量化**
   - `3_1_tutorial_vectors.ipynb`
   - 依赖 Chronos 模型路径 `CHRONOS_MODEL_PATH`
   - 输出目录：
     - `vector_db_w192_s6/query/`
     - `vector_db_w192_s6/target/`
   - 文件包括：
     - `embeddings.npy`
     - `id_mapping.pkl`
     - `meta.json`
     - `faiss_index.bin`
     - `metadata.json`

4. **可视化（可选）**
   - `3_2_tutorial_visualize.ipynb`
   - 生成：
     - `annual_mean_temperature.png`
     - `monthly_climatology.png`
     - `<city>_time_series.png`
     - `random_8day_hourly_*.png`

5. **RAG 检索可视化**
   - `4_1_tutorial_rag.ipynb`
   - 输出：`results_demo/query*_top*.png`

6. **批量 RAG 数据集生成**
   - `4_2_tutorial_rag_process.ipynb`
   - 输出：
     - `rag_results_all.npz`
     - `rag_results_all.json`
     - `rag_results_all_train/val/test.npz`

## 常见问题

- **路径找不到文件**：优先检查各 notebook 的 `*_DIR` 是否指向同一数据根目录。
- **Chronos 模型加载失败**：确认 `CHRONOS_MODEL_PATH` 指向本地模型目录。
- **FAISS 报错**：CPU 用 `faiss-cpu`，GPU 用 `faiss-gpu`。
