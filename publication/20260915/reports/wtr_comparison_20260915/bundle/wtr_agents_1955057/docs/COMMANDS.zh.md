# 实验与部署命令

以下是可执行命令模板，不是已经部署的回执。路径通过现有服务器资产映射填写，不能把本地 `/mnt/data` 当训练服务器路径。仅第一部分和CLI/CPU单测可在无GPU环境执行。

## 1. 固定隔离工作树

```bash
set -euo pipefail
: "${REPO:?设置现有 BMCR-T-AdaTAD git 仓库绝对路径}"
: "${WORKTREE:?设置新的只读基线工作树绝对路径，路径必须尚不存在}"
: "${BUNDLE:?设置解压后 wtr_agents_1955057 的绝对路径}"
: "${RESOURCES:?设置现有 research/paper/resources.local.json 的绝对路径}"
: "${OUT:?设置本轮独立结果目录绝对路径}"
BASE=1955057508af5a5dfd59a98bddf49302bee5972c

git -C "$REPO" fetch origin codex/graph-tad-20260914
git -C "$REPO" worktree add --detach "$WORKTREE" "$BASE"
test "$(git -C "$WORKTREE" rev-parse HEAD)" = "$BASE"
export PYTHONPATH="$BUNDLE:$WORKTREE:$WORKTREE/upstream:${PYTHONPATH:-}"
mkdir -p "$OUT"
python -m pytest -q "$BUNDLE/tests"
python -m compileall -q "$BUNDLE/wtr"
python -m wtr.snapshot_metrics --repo "$WORKTREE" --output "$OUT/snapshot_metrics.json"
```

这里不包含下载模型或数据；资源清单的本地绝对路径必须已由现场资产检查确认。

## 2. 核对原模型命令接口，不训练

```bash
CFG="$WORKTREE/configs/paper_review/review5485_full_v2_s_seed42.json"
python "$WORKTREE/tools/paper_train.py" --config "$CFG" --resources "$RESOURCES" --dry-run
python "$WORKTREE/tools/paper_eval.py" --help
python -m wtr.probe --help
python -m wtr.fit_router --help
```

新方法不能借 `--dry-run` 的成功冒充真实初始化、GPU检查和mAP。接收agent必须首先审查 integration.py 与这台服务器 OpenTAD API 是否完全一致。

## 3. 建立 K384/D100/S100 的第一份不可变动作库

使用同一V2课程真实存在的20/40/60轮checkpoint，分别命名：

```bash
: "${CKPT_ANCHOR:?较早checkpoint，例如同一课程epoch020}"
: "${CKPT_CURRENT:?当前checkpoint，例如同一课程epoch040}"
: "${CKPT_LATER:?更晚checkpoint，例如同一课程epoch060；仅作为诊断真值}"
```

必须来自同一cfg、相同训练长度定义，不能跨V1/V2或者不同seed相减。`learned` 与 `ema` 不混用。

先运行6视频技术版：

```bash
python -m wtr.probe build \
  --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
  --checkpoint "$CKPT_CURRENT" --state learned \
  --plans 1 --selector anchor --scope local --epoch 0 \
  --max-videos 6 --pairs 2 --output "$OUT/smoke_bank.pt"

python -m wtr.probe replay \
  --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
  --checkpoint "$CKPT_CURRENT" --state learned \
  --bank "$OUT/smoke_bank.pt" --kind frame --measured \
  --output "$OUT/smoke_values.jsonl"
```

重复重放需换输出路径；工具拒绝覆盖。检验同action重复差接近数值误差，原模型state/buffers不变、实际RGB重编码、没有GT泄漏到路由输入，然后建立32视频版本：

```bash
python -m wtr.probe build \
  --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
  --checkpoint "$CKPT_CURRENT" --state learned \
  --plans 1 --selector anchor --scope local --epoch 0 \
  --max-videos 32 --pairs 8 --output "$OUT/bank.pt"

for NAME in anchor current later; do
  case "$NAME" in
    anchor) CKPT="$CKPT_ANCHOR" ;;
    current) CKPT="$CKPT_CURRENT" ;;
    later) CKPT="$CKPT_LATER" ;;
  esac
  python -m wtr.probe replay \
    --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
    --checkpoint "$CKPT" --state learned --bank "$OUT/bank.pt" \
    --kind frame --output "$OUT/$NAME.jsonl"
done

# genuine EMA control：同一 current checkpoint 的 ema 状态在相同动作库上重执行
python -m wtr.probe replay \
  --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
  --checkpoint "$CKPT_CURRENT" --state ema --bank "$OUT/bank.pt" \
  --kind frame --output "$OUT/current_ema.jsonl"
```

32视频是资源受限pilot，不是最终全数据集统计。最终需扩充为按类别/时长/背景比例覆盖的train-development划分，且保留独立video holdout。

## 4. 算子损害解耦，不改变网络源代码

```bash
python -m wtr.probe replay \
  --repo "$WORKTREE" --config "$CFG" --resources "$RESOURCES" \
  --checkpoint "$CKPT_CURRENT" --state learned --bank "$OUT/bank.pt" \
  --kind operators --operator-plan 1 --groups 1 --measured \
  --output "$OUT/operator_values.jsonl"
```

该工具固定选帧与所有非目标route masks，比较一个2×2 token组的 attention-hold、FFN-light 及两者联合。`branch_parity_max_abs`不通过，先修实验实现，不解释效用。

负 `actual_delta` 表示这次删减伤害任务；`removal_cost` 是相反数。若要做购买heavy计算的value，应从cheap状态出发重新执行，不能机械把dense-removal值当所有稀疏状态的增益。

## 5. 未来价值预测：只做诊断，不使用未来文件训练

```bash
python -m wtr.forecast \
  --anchor "$OUT/anchor.jsonl" --current "$OUT/current.jsonl" \
  --future "$OUT/later.jsonl" --ema "$OUT/current_ema.jsonl" \
  --betas 1,1.025,1.05,1.1,1.2 --split holdout --bootstrap 2000 \
  --output "$OUT/forecast_holdout.json"
```

输出Spearman、真实后续value下的regret、top1一致率、MSE及video-cluster CI。不是mAP。不能在holdout上选最好β后又将同一holdout当最终确认；β扫描本身是机制报告，正式选择需另用calibration并在新holdout验证。

## 6. 冻结detector的 FrameRouter 训练对照

```bash
for MODE in none current ema future; do
  EXTRA=()
  if [[ "$MODE" == "future" ]]; then EXTRA+=(--anchor "$OUT/anchor.jsonl"); fi
  if [[ "$MODE" == "ema" ]]; then EXTRA+=(--ema "$OUT/current_ema.jsonl"); fi
  python -m wtr.fit_router \
    --repo "$WORKTREE" --checkpoint "$CKPT_CURRENT" --state learned \
    --current "$OUT/current.jsonl" --mode "$MODE" "${EXTRA[@]}" \
    --beta 1.1 --steps 2000 --seed 42 --lr 1e-4 --lambda-js .1 \
    --output "$OUT/router_${MODE}_eval_only.pth"
done
```

所有模式保留current实际GT价值NLL，仅辅助排序分布teacher不同。`none`无额外JSD；`current`是current-value JSD，不是原模型对自身同输入零梯度KL；`ema`是真实EMA重执行值；`future`从earlier/current真实价值构造teacher。由此首先测“未来价值方向是否值得利用”，未宣称已经免除反事实标签成本或实现完整在线递归。

导出的`ema`和`learned`都指向同一evaluation-only模型，不能误写成额外EMA集成增益。禁止把这些文件给原 `paper_train --resume`，它们没有原optimizer/scheduler状态。

## 7. 冻结方案后完整测试

```bash
# 同一 current detector 初始 FrameRouter 对照
python "$WORKTREE/tools/paper_eval.py" \
  --config "$CFG" --resources "$RESOURCES" --checkpoint "$CKPT_CURRENT" \
  --state learned --force-plan 1 --output "$OUT/eval_original_refiner"

for MODE in none current ema future; do
  python "$WORKTREE/tools/paper_eval.py" \
    --config "$CFG" --resources "$RESOURCES" \
    --checkpoint "$OUT/router_${MODE}_eval_only.pth" \
    --state learned --force-plan 1 --output "$OUT/eval_router_${MODE}"
done

# 同checkpoint的uniform推理控制；不是重新训练后的Uniform Full
python "$WORKTREE/tools/paper_eval.py" \
  --config "$CFG" --resources "$RESOURCES" --checkpoint "$CKPT_CURRENT" \
  --state learned --force-plan 1 --selector uniform --disable-frame \
  --output "$OUT/eval_uniform_same_checkpoint"
```

完整测试必须核对211视频与792窗口及官方AP。原配置的训练数据、官方test此前已参与过开发比较，需如实披露model-selection流程；不能声称新设一个holdout就让过去测试重新独立。

## 8. 生成原仓库已经能执行的控制配置

```bash
python -m wtr.make_configs --base "$CFG" \
  --output "$OUT/configs_s" --epochs 80 --schedule-epochs 80

# 这里只展示一条；必须由唯一deployment owner批准进入队列
NEWCFG="$OUT/configs_s/wtr_fixed_k384_full_ds_s_seed42.json"
python "$WORKTREE/tools/paper_train.py" --config "$NEWCFG" \
  --resources "$RESOURCES" --output "$OUT/preflight/wtr_fixed_k384_full_ds_s_seed42" --preflight

# preflight与正式目录必须分离，不能直接resume技术测试
python "$WORKTREE/tools/paper_train.py" --config "$NEWCFG" \
  --resources "$RESOURCES" --output "$OUT/train/wtr_fixed_k384_full_ds_s_seed42" \
  --slice-hours 10

# 收到exit75且checkpoint真实存在后，原控制器用同一配置续跑
python "$WORKTREE/tools/paper_train.py" --config "$NEWCFG" \
  --resources "$RESOURCES" --output "$OUT/train/wtr_fixed_k384_full_ds_s_seed42" \
  --slice-hours 10 --resume
```

这些不是新增完整WTR模型，仅为现有schema的固定计划/均匀掩码/plan-aware控制。20轮筛查必须使用固定80轮LR horizon，且不能将20轮终止的配置改为80轮然后加载旧scheduler假称无缝续跑。推荐80轮config+原milestone，是否新增更多课程另行审批。

## 9. Slurm与唯一控制器

原源码使用 `--partition=gpu --qos=gpugpu --gres=gpu:1 --cpus-per-task=6 --signal=B:USR1@600`；现场仍需核验这些站点配置和允许节点。不能凭旧PID、旧节点表推定现在资源空闲。

不建议启动新的 `paper_dispatch --submit`。owner把上面的程序放入现有队列中新stage：
```
args[0] = /已冻结bundle/dispatch_entry.py
args[1:] = ["probe", "replay", "--repo", ...]
```
`dispatch_entry.py`只是单次python入口，不是第二个控制器。

依赖图建议：
```
WTR_CPU_CONTRACT
 -> WTR_GPU_SMOKE
 -> WTR_BANK_BUILD
 -> {WTR_REPLAY_ANCHOR, WTR_REPLAY_CURRENT, WTR_REPLAY_LATER, WTR_REPLAY_EMA,
     WTR_OPERATOR_PROBE}
 -> WTR_FORECAST
 -> {WTR_FIT_NONE, WTR_FIT_CURRENT, WTR_FIT_EMA, WTR_FIT_FUTURE}
 -> WTR_LOCK_PROTOCOL
 -> WTR_FULL_EVAL
```

首批新增同时最多2个GPU诊断是建议上限，不替代现有账户/owner更严格配额。CPU分析不排GPU。不要为腾资源取消FullV2、Uniform、Native384等关键已提交课程。不是所有节点完成后都自动扩展80轮实验：技术验收与科学晋级应分开，后者由研究者批准。

## 10. 区分“适合当前detector”与“适合更成熟detector”

未来value可能与当前detector不匹配，预测相关性不能代替部署收益。另做policy-transfer，只转移已训练的frame_router，保留later detector全部其他参数，仍不使用later真实value标签训练：

```bash
for MODE in none current ema future; do
  python -m wtr.transfer_router \
    --router-checkpoint "$OUT/router_${MODE}_eval_only.pth" \
    --target-checkpoint "$CKPT_LATER" --target-state learned \
    --output "$OUT/router_${MODE}_on_later_eval_only.pth"
  # 官方test只在方案冻结后运行；前期应由agent补充video-disjoint development evaluator
  python "$WORKTREE/tools/paper_eval.py" \
    --config "$CFG" --resources "$RESOURCES" \
    --checkpoint "$OUT/router_${MODE}_on_later_eval_only.pth" \
    --state learned --force-plan 1 --output "$OUT/eval_${MODE}_on_later"
done
```

此实验仍不是在线递归训练；它分离了teacher目标预测能力与scout/representation漂移。若未来teacher只在离线相关性提高、在实际当前/后期TAD均无收益，应停止将其列为已验证贡献。

测量补充：bank保存current point-head的loss normalizer；所有checkpoint探针暂时使用同一分母并恢复原状态，避免teacher比较混入loss归一化漂移。
