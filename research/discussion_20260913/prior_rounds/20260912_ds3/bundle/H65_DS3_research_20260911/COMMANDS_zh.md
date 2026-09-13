# H65-DS3 agents 执行命令与实验启动顺序

## 0. 命令的三种状态

**可立即使用的执行包命令**：`bin/bootstrap.sh`、`bin/source_audit.py`、`bin/run_agent.sh`、`bin/run_experiment.py` 和 `bin/verify_bundle.py` 已随本包提供。前两者只审计/创建隔离worktree；agent命令需要已有Codex CLI；experiment默认仅打印计划。

**已核实的仓库命令**：cycle4 `.sbatch`、`tools/train.py`、`tools/test.py` 与已有focused tests都来自固定提交。服务器路径、checkpoint和GPU allocation仍需用户环境实际满足。

**必须先由agents实现的命令**：`tools/bata/ds3_validate.py`、`ds3_profile.py` 和 `configs/adatad/thumos/ds3/*.py`。接口在 `contracts/cli_v1.md` 中定义，当前H65提交并不存在；以下命令不能提前运行。

本包没有在服务器上执行训练、TAD评估或Codex任务。不要把下载本包等同于模型已实现。

## 1. 建立隔离分支，保留当前工作

将执行包解压到已有授权工作区。以下示例使用现有仓库记载的远端根目录；在本机运行时改为本机绝对路径。只需要修改前三个路径变量。脚本不会clone、fetch、覆盖目录、reset或清理你的工作树。

```bash
BASE_ROOT=/data/run01/sczc063/yuzibo
export BUNDLE="$BASE_ROOT/H65_DS3_research_20260911"
export REPO="$BASE_ROOT/OpenTAD_C3_CoarseClean_20260702"
STAMP="$(date +%Y%m%d-%H%M%S)"
export WORKTREE="$BASE_ROOT/worktrees/h65_ds3_$STAMP"
export RUN_ROOT="$BASE_ROOT/experiments/h65_ds3_$STAMP"

python "$BUNDLE/bin/verify_bundle.py"
bash "$BUNDLE/bin/bootstrap.sh" "$REPO" "$WORKTREE" "$RUN_ROOT"
source "$RUN_ROOT/session.env"
python "$BUNDLE/bin/source_audit.py" \
  --repo "$WORKTREE" --out "$RUN_ROOT/audit/source_recheck.json"
```

若本地没有anchor对象，bootstrap会停止。应先通过已有授权仓库连接获取并核实确切提交；不要改用默认分支绕过。若用户已有较新本地实现，A01比较差异并记录，不能静默忽略或混入。

## 2. agents 顺序执行

命令使用 `codex exec --sandbox workspace-write`、stdin prompt和`--output-last-message`；启动器先检查本机CLI帮助。不安装、不切换模型、不绕过sandbox。也可将对应Markdown任务书直接粘贴到既有coding-agent会话。[R28]

```bash
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 00_coordinator.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 01_audit.md
```

读完 `RUN_ROOT/agents` 中这两项输出并解决来源问题，再继续实现：

```bash
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 02_time_frontend.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 03_dense_aux_depth.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 04_spatial_runtime.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 05_integration_training.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 06_eval_profile.md
bash "$BUNDLE/bin/run_agent.sh" "$WORKTREE" "$RUN_ROOT" 07_independent_review.md
```

共享worktree不要把这些命令追加 `&` 并行执行。A02/A03/A04可在独立worktree并行，但共享核心文件只由A05集成；人工审查各分支commit后再cherry-pick，不让脚本自动解决冲突。最后A07给出是否允许进入GPU smoke的明确结论。

为了先完成最重要的时间基线，可以在A02后先让A05只集成D768/Z0，再让A06/A07检查；空间模块不应阻塞第一批Z0实验。

## 3. 现有 H65 cycle4 的基准检查

这部分只验证历史H65环境和入口，**不是新dense768 teacher训练**。在现有OpenTAD环境、允许的计算节点中执行。Slurm环境保持调度器给的CUDA_VISIBLE_DEVICES，不强制改成物理GPU1；旧指定GPU1子脚本另按其规则执行。[C01,C11]

```bash
cd "$WORKTREE"
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

python -m py_compile tools/train.py tools/test.py \
  tools/bata/validate_duca_h65_first_singleclock_cycle4.py
python -m pytest tests/test_duca_h65_cycle4_singleclock_contract.py -q
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py \
  tests/test_c3_asformer_delta_ledger_full_train.py -q

MODE=STAGE1 PRECHECK_ONLY=1 PRE_RUN_ONLY=0 \
DUCA_REPO_ROOT="$WORKTREE" \
bash scripts/run_duca_h65_matched_cycle4_n16r4.sbatch
```

需要检查cycle4 stage2时，必须提供真实且已核实的stage1 checkpoint。以下变量不能指向dense768或历史20epoch checkpoint：

```bash
: "${DUCA_STAGE1_CHECKPOINT:?Set the actual cycle4 uniform384 epoch29 checkpoint}"
export DUCA_STAGE1_CHECKPOINT_SHA256="$(sha256sum "$DUCA_STAGE1_CHECKPOINT" | awk '{print $1}')"
export DUCA_STAGE1_CHECKPOINT_EPOCH=29
MODE=STAGE2_ON PRECHECK_TARGET=STAGE2_ON PRECHECK_ONLY=1 PRE_RUN_ONLY=0 \
DUCA_REPO_ROOT="$WORKTREE" \
bash scripts/run_duca_h65_matched_cycle4_n16r4.sbatch
```

只有明确需要历史路径GPU smoke时才提交下面作业，不能在登录节点直接跑PRE_RUN。日志放外部目录，正式训练预算不被改成1：[C01]

```bash
mkdir -p "$RUN_ROOT/legacy_smoke"
sbatch --chdir="$WORKTREE" \
  --output="$RUN_ROOT/legacy_smoke/%j.out" \
  --error="$RUN_ROOT/legacy_smoke/%j.err" \
  --export=ALL,DUCA_REPO_ROOT="$WORKTREE",MODE=STAGE1,PRECHECK_ONLY=0,PRE_RUN_ONLY=1,DUCA_PRE_RUN_MAX_ITERS=2,DUCA_PRE_RUN_WORK_DIR="$RUN_ROOT/legacy_smoke/work" \
  "$WORKTREE/scripts/run_duca_h65_matched_cycle4_n16r4.sbatch"
```

两迭代后仍需看成功update、非零LR和参数delta。该命令不代表stage1训练收敛，也不产生可冒充dense teacher的checkpoint。

## 4. 新模型完成后：CPU检查、来源清单与CUDA门禁

以下从本节起依赖agents新建的模型与CLI。先检查文件真实存在，并只使用新研究分支的reviewed commit。

```bash
cd "$WORKTREE"
test -f tools/bata/ds3_validate.py
test -f tools/bata/ds3_profile.py
test -f configs/adatad/thumos/ds3/dense768.py
test -f configs/adatad/thumos/ds3/z0_uniform24.py

python tools/bata/ds3_validate.py --help
python tools/bata/ds3_profile.py --help
python -m pytest tests -k 'ds3' -q  # 必须确认collected>0，非“零测试成功”
```

A05根据 `templates/launch.template.json` 创建真实manifest，不能保留占位path/hash。新config的训练初始化写resolved config，不给旧train.py发明`--teacher`等CLI。审核文件至少包括D768教师来源、Z24推理以及D1训练三份独立manifest。

```bash
export MANIFEST="$RUN_ROOT/manifests/Z24.json"
python "$BUNDLE/bin/run_experiment.py" "$MANIFEST"   # 仅计划；缺gate会列出issues

python tools/bata/ds3_validate.py source --manifest "$MANIFEST" --out "$RUN_ROOT/audit/source.json"
python tools/bata/ds3_validate.py config --manifest "$MANIFEST" --out "$RUN_ROOT/audit/config.json"
python tools/bata/ds3_validate.py checkpoint --manifest "$MANIFEST" --out "$RUN_ROOT/audit/checkpoint.json"
python tools/bata/ds3_validate.py gates --manifest "$MANIFEST" --gate G0 --out "$RUN_ROOT/audit/G0.json"
```

G1/G2应在已分配的计算节点中、经过A07同意的界定验证执行。先将manifest状态批准为`APPROVED_FOR_VALIDATION`，允许短验证但不是正式全量训练。以下不会自动获得allocation；没有allocation就停止：

```bash
# 仅在现有Slurm allocation内设置；不要在login节点用这个变量伪装授权。
export H65_DS3_ALLOW_GPU_RUN=1
python tools/bata/ds3_validate.py gates --manifest "$MANIFEST" \
  --gate G1 --out "$RUN_ROOT/audit/G1.json" --allow-gpu

# G2使用独立D1辅助训练manifest，至少2次成功非零LR更新。
python tools/bata/ds3_validate.py gates --manifest "$RUN_ROOT/manifests/AUX_smoke.json" \
  --gate G2 --out "$RUN_ROOT/audit/G2.json" --allow-gpu
```

没有真实dense768 checkpoint时，G0会把该依赖标BLOCKED；先完成新D768训练配置的来源/数据/优化器检查及其对应smoke，在新dense训练manifest中显式记录“from-pretrain dense teacher”。不能用unknown checkpoint绕过。训练D768和验证D768使用不同manifest，前者不要求已存在最终dense checkpoint。

## 5. 正式运行：按阶段单独批准，不一键全矩阵

新代码与配置提交后，manifest的 `expected_commit` 应写实际reviewed DS3 commit，而不是永远写anchor；该commit必须是anchor的后代。配置、checkpoint、gate证据、resolved config和数据split哈希全部填入。A07审查后由负责人设为`APPROVED_FOR_THIS_RUN`。模板中的状态默认不批准，启动器不会自动替你批准。

```bash
python "$BUNDLE/bin/run_experiment.py" "$RUN_ROOT/manifests/Z24.json"
```

确认计划issues为空，再提交已批准的单个作业。`templates/run_ds3.sbatch` 的partition/account/time基于当前仓库示例，但应按站点实际策略审查；不含任何凭证：

```bash
mkdir -p "$RUN_ROOT/slurm"
export BUNDLE
export MANIFEST="$RUN_ROOT/manifests/Z24.json"
export H65_DS3_ALLOW_GPU_RUN=1
sbatch --output="$RUN_ROOT/slurm/%j.out" --error="$RUN_ROOT/slurm/%j.err" \
  --export=ALL "$BUNDLE/templates/run_ds3.sbatch"
```

在已有合法compute allocation中，也可直接运行同一受控启动器：

```bash
python "$BUNDLE/bin/run_experiment.py" "$MANIFEST" --execute
```

若不是Slurm但确为站点批准的保护分配，需要负责人显式设置 `H65_DS3_COMPUTE_ALLOCATION_CONFIRMED=1`；不能为绕过login限制而设置。脚本只执行一个已封存的train/eval，不自动串联实验，不覆盖已有非空输出目录。

推荐提交顺序：D768可复现 → Z36/Z24/Z16/ZR24 → AUX → PONLY/T24U/T24A → D8/DAD → S75/S50 →温和预算因子矩阵 → CAL →必要时JOINT。训练aux的预算一致，最终模型用三seed；不要一开始就跑全三维强剪枝网格。

## 6. 性能测量与验收

只对已过身份、泄漏、来源和执行计数门禁的模型测实际速度。三种scope分开输出：[参见拟新增CLI合同]

```bash
python tools/bata/ds3_profile.py --manifest "$MANIFEST" \
  --scope backbone --warmup 20 --iterations 100 --out "$RUN_ROOT/outputs/Z24.backbone.json" --allow-gpu
python tools/bata/ds3_profile.py --manifest "$MANIFEST" \
  --scope model --warmup 20 --iterations 100 --out "$RUN_ROOT/outputs/Z24.model.json" --allow-gpu
python tools/bata/ds3_profile.py --manifest "$MANIFEST" \
  --scope end_to_end --warmup 20 --iterations 100 --out "$RUN_ROOT/outputs/Z24.e2e.json" --allow-gpu
```

输出mAP和时延前再次运行A07，提供真实的logs/metrics/profiler/gates路径，不让它只审理论FLOPs。最终表格必须有NOT_RUN/null，而不是用0填未知性能。

## 7. 可直接粘贴给主agent的总指令

```text
以研究包REPORT_zh.md为方法规格，代码锚点04c35a3b76897e6c1569eeede41ed3aecaf7f854。
先只做A01源码和dense teacher来源审计，再实现P1完整original-clip采样及native384恢复。
不要重写历史H65，不混入BMCR仓库或默认分支，不把uniform384叫dense768。
严格区分Z0、D1、C2、J3。主方法D1训练时全部clips、patches、layers完整前向；
训练preview最终特征H0、浅层出口H8，推理才执行cheap/8/12，检测器仍吃dense768。
空间先只稀疏heavy MLP，attention与TIA保持dense；更激进merge另做对照。
所有新CLI遵循contracts/cli_v1.md；全部new paths是待实现，不得假装现成。
先通过identity、坐标、checkpoint、冻结/非零LR、实际compact和无teacher/cache泄漏测试。
本轮不启动GPU或Slurm，不升级环境，不push，不输出凭证；交付实现、CPU测试、待运行命令与阻塞项。
没有真实证据的结果标NOT_RUN。每阶段按agents/任务分工提交，未经审查不自动进入后续重训练。
```
