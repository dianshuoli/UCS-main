以下是一个简洁的 `README.md` 示例，用于介绍推理代码的运行方式和模型权重的下载地址：

---

# Inference Guide

This repository provides inference code for running predictions using the trained model.

## 🔧 Inference Command

```bash
python test.py \
  --work_dir <work dir> \
  --run_name <result path name> \
  --maskResultDir <result dir name> \
  --data_path <test data path> \
  --sam_checkpoint ./workdir/models/UCS_L_best.pth
```

Replace the arguments with your own paths:

* `--work_dir`: Directory for logs and temporary files
* `--run_name`: Name for saving result files
* `--maskResultDir`: Directory to store prediction results
* `--data_path`: Path to your test dataset
* `--sam_checkpoint`: Path to the model checkpoint

## 📥 Model Checkpoint

You can download the pretrained model checkpoint from [here](https://drive.google.com/file/d/1zMeyKzm8uJlCoLL6augRQY5oeSDyKGcO/view?usp=sharing) and place it in:

```bash
./workdir/models/UCS_L_best.pth
```

---
