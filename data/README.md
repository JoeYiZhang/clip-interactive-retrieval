# 数据目录说明

本目录只保存数据清单、校验值和下载说明，不提交公开数据集原图、用户反馈原文或模型权重。原图、嵌入与索引统一放在 `D:\image_data`。

当前接入：

- CIRR：rc2 官方公开标注与划分已就绪，原图等待官方/NLVR2 回复。
- FashionIQ：官方标注已就绪；用户批准的 HF 镜像归档 SHA256 已核对，train/val 图片缺失 0，验证集 B0-B3 已完成。
- 可选 CIRCO：存在多个相关目标时使用 mAP@K。

```text
D:\image_data\
  datasets\cirr\{captions,image_splits,img_raw}
  datasets\fashioniq\{captions,image_splits,images}
  downloads\
  derived\{embeddings,indexes,rankings}
  manifests\
```

接入任何完整数据前必须在 `manifests/` 中记录：来源链接、许可、版本、文件数量、校验值、划分方式和本地路径。CIRR 原图未获回复前不得用第三方镜像绕过申请；FashionIQ 第三方镜像已获用户确认，报告必须披露来源。用户反馈数据必须去标识化，并按用户或会话分组切分，避免泄漏。

`scripts/validate_dataset.py` 中 `structural_ok=true` 只表示标注关系正确；只有 `ready_for_experiment=true` 才表示该划分可运行正式实验。

