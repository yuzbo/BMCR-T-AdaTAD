# 命令状态与运行顺序

## 已提供、可直接运行的本地命令

以下命令只测试本包独立CPU机制、验证计划或绘制历史记录，不会训练VideoMAE，也不会提交Slurm。需要已安装Python、PyTorch、Matplotlib；不自动升级项目旧环境。

```bash
cd /absolute/path/to/tad3d_agents
python scripts/run_reference_checks.py
python scripts/check_plan.py
python scripts/plot_results.py --input data/historical_results.csv --out figures/historical
```

准备固定项目工作树（只操作指定新目录，不覆盖原目录）：

```bash
bash scripts/prepare_worktree.sh \
  /absolute/path/to/BMCR-T-AdaTAD \
  /absolute/path/to/BMCR-frame-writeback
cd /absolute/path/to/BMCR-frame-writeback
git switch -c research/frame-writeback-239d098
```

将`AGENTS_TASKS.zh.md`中的总控任务交给agent，再按A–G角色分工。不存在本包假设已安装的通用`agents run`命令；不要把未验证CLI当成已可执行工具。

## 以下是 agents 必须实现的生产CLI合同，不是当前仓库已有命令

所有生产工具先实现`--help`、`--dry-run`和本地测试，通过后才可进入GPU执行。`resources.local.json`由真实资源清单生成，不在本包虚构checkpoint路径。

```bash
# Agent A/E: implement first; these commands do not exist in the fixed snapshot.
python tools/frame_audit.py --base-commit 239d098cd899936c35989fae6243c70259a85adb \
  --resources research/frame/resources.local.json --output research/frame/audit.json

python tools/frame_plan.py --experiments research/frame/experiments.json \
  --resources research/frame/resources.local.json --dry-run

# Native/legacy/teacher/gate tests (agent-created tests, distinct from this package's toy tests).
python -m pytest tests/frame/test_legacy_identity.py tests/frame/test_geometry.py \
  tests/frame/test_decoder.py tests/frame/test_engine.py -q

# Stage R: fixed existing selector, K384, frozen encoder, original-grid recovery.
python tools/frame_train.py --config configs/frame/s_recovery_k384.py \
  --resources research/frame/resources.local.json --dry-run

# Stage U: distinguish representation-repair from actual same-K intervention.
python tools/frame_intervene.py --config configs/frame/s_intervention_k384.py \
  --resources research/frame/resources.local.json --dry-run

# Stage DS: native-state heavy query/FFN, full spatial TIA state.
python tools/frame_train.py --config configs/frame/s_joint_3d.py \
  --resources research/frame/resources.local.json --dry-run

# Complete evaluation / matched-profile / deterministic analysis.
python tools/frame_eval.py --config configs/frame/s_joint_3d.py \
  --checkpoint /actual/verified/checkpoint.pth --dry-run
python tools/frame_profile.py --config configs/frame/s_joint_3d.py \
  --checkpoint /actual/verified/checkpoint.pth --cases full,partial,short --dry-run
python tools/frame_analyze.py --manifest research/frame/completed_manifest.json \
  --figspec research/frame/figures.csv --output paper/analysis --dry-run

# Default dispatch writes scripts/manifest only; implement explicit authorization handling.
python tools/frame_dispatch.py --plan research/frame/plan.json --emit-only
```

**执行界限**：删掉`--dry-run`不是自动得到授权。先有当前资源核实与用户明确执行授权，再由现场调度器提交OWN作业。历史T24实验归档且不进入新计划；是否停止任何仍运行的旧作业要读取当前队列、保存checkpoint并另行确认，禁止套用旧snapshot job IDs取消作业。

## 配置必须明确记录的字段

backbone、anchor_checkpoint及配方、external_teacher_checkpoint、teacher_kind、decoder_arch/init、feature_space、K、temporal_selection_unit、native_length、detector_length、TIA_temporal_length、dense/sparse训练图、trainable_groups、所有loss与权重、successful_updates预算、随机种子、数据ID、precision、time coordinate unit、checkpoint_selection。

原baseline warm/joint命令已经存在，但不要从本包直接启动；Agent A按固定源码核实资源与审计输入后，复用原`tools/full_train.py`流程完成B01。新frame训练不能隐式继承旧recipe resume状态。
