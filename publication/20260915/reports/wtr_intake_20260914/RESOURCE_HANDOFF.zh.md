# 新服务器资源交接记录

来源：本轮从任务 `01a0a03b-8d1a-7f91-81b9-33c4fd8b0314` 收到的资源交接消息。本文件记录该任务报告的验证结果；本轮没有重新登录服务器核验，也没有启动训练。

- 连接命令：`ssh -p 44909 root@connect.nmb1.seetacloud.com`。
- 数据：`/root/autodl-tmp/thumos14/`，train/含200视频，test/含211视频，annotations/含thumos_14_anno.json、category_idx.txt、tad_train_videos.txt、tad_test_videos.txt等。总量约34.68GB，源端rsync缺失文件及文件大小核验通过。目标SYNC_STATUS.txt为`COMPLETE_200_train_211_test_annotations_verified_by_file_size`。
- OpenTAD：`/root/autodl-tmp/OpenTAD`，提交`346d09d19e2091372cec48172dbe40f7b28bdee6`。
- Conda环境：`/root/autodl-tmp/envs/opentad`；Python：`/root/autodl-tmp/envs/opentad/bin/python`。`source /root/autodl-tmp/opentad_setup/activate.sh`激活并进入源码目录。
- 环境版本：Python3.10.12、PyTorch2.0.1+cu118、Torchvision0.15.2+cu118、MMCV2.0.1、MMAction2 1.1.0、NumPy1.23.5、Decord0.6.0；CUDA toolkit `/usr/local/cuda-11.8`。
- 交接报告已验证：两张RTX4080 SUPER计算、Align1D/边界池化前后向、MMCV NMS、OpenTAD导入、CPU NMS/Soft-NMS、训练测试入口、实际视频解码。
- Decord安装包有旧平台标签的pip check告警；交接报告导入和解码通过。
- 文档与日志：`/root/autodl-tmp/opentad_setup/README.md`、`pip-freeze.txt`、`verify.log`。
- 交接状态：尚未启动训练。

本消息未确认H65/BMCR/Scout、Cross/R03、任务teacher/head、诊断checkpoint或WTR补丁已同步。它也不证明旧Slurm实验队列已迁移到此服务器。
