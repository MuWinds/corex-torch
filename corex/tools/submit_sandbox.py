"""提交一个「常驻沙箱」作业：占住一台节点的 8 张卡，循环跑队列里的脚本。

为什么要有这个工具：沙箱作业有**寿命**，到点容器就没了，队列里没跑完的东西全部作废。
同一机箱里每次卡死的是哪张、哪个节点更健康，都是**会话级**事实，不能按节点名判断；
所以作业必须**钉节点**——REST 接口没有节点约束字段，只能走 ``cbatch -w NODE``。
``platform/scow/tools/run_pinned.py`` 能做这件事，但它**会一直等到作业结束**
（对 4320 分钟上限的作业等于挂住 3 天），所以这里只发不等。

一次发两个作业的用法：第一个跑完、容器消失后，第二个被调度起来接着吃同一个队列目录
``coffin-work/base/sandbox``，于是**队列跨容器连续**，中间不留空档。
作业时长上限是平台 QoS 的 4320 分钟（3 天），超过会被 500 拒掉。

用法（本机，能连 SCOW 的登录节点）::

    python tools/submit_sandbox.py --name corex-a
    python tools/submit_sandbox.py --name corex-b

Constraints:
    * ``command`` 只能是一行，所以载荷是 base64 编码后塞进 ``cbatch --wrap``；
    * 脚本内容不能含字面量 ``#!/bin/bash``（SCOW 网关会把它过滤成 0 字节）；
    * 必须带 ``--gres gpu:8``：不带的话作业不持有加速卡，``torch.cuda`` 会报
      「No CUDA GPUs are available」——看起来像卡坏了，其实只是没申请；
    * 8 卡作业必须显式给 ``coreCount``（门户默认 14*8=112 超过单节点 110 核）。
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import sys
import time

#: 仓库根的上一级是 coffin-repo，platform/scow 在那里。
_PLATFORM = pathlib.Path(__file__).resolve().parents[3] / "platform"
sys.path.insert(0, str(_PLATFORM))

from scow import ScowClient  # noqa: E402

#: 远程 runner 脚本的来源（沙箱的唯一运行器，队列协议只此一份）。
RUNNER_SRC = pathlib.Path(__file__).resolve().parents[3] / "corex_drive_re" / "sandbox" / "sandbox.sh"


def main(api: argparse.Namespace) -> int:
    """提交一个沙箱作业并打印它的 id。

    Args:
        api: 命令行参数（名字、节点、卡数、分钟数、沙箱根目录）。

    Returns:
        进程退出码。

    Raises:
        无：作业提交失败只返回非 0，不抛异常——调用方常常要连续发两次。
    """
    home = api.home
    base = f"{home}/coffin-work/base"
    tools_dir = f"{home}/coffin-work/tools"
    remote_runner = f"{tools_dir}/sandbox.sh"

    client = ScowClient()
    client.run(f"mkdir -p {tools_dir}", timeout=60)
    # 上传前把首行的 shebang 去掉：SCOW 网关按内容过滤，带 `#!` 的文本会被降级成 0 字节或直接
    # 403（AGENTS.md 里记过同一类坑）。runner 是用 `bash <file>` 起来的，shebang 本来也没用。
    runner_text = RUNNER_SRC.read_text(encoding="utf-8")
    lines = runner_text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    staged = pathlib.Path(__file__).resolve().parent / ".runner.tmp.sh"
    staged.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    try:
        client.upload(staged, remote_runner)
    finally:
        staged.unlink(missing_ok=True)

    launcher = "\n".join([
        "set -u",
        f"export SANDBOX_ROOT={home}/{api.sandbox}",
        f"export LIFETIME_HOURS={api.hours}",
        f"export SCRIPT_TIMEOUT={api.script_timeout}",
        f"bash {remote_runner}",
        "",
    ])
    remote_launcher = f"{tools_dir}/{api.name}.sh"
    local_launcher = pathlib.Path(__file__).resolve().parent / f".{api.name}.tmp.sh"
    local_launcher.write_text(launcher, encoding="utf-8", newline="\n")
    try:
        client.upload(local_launcher, remote_launcher)
    finally:
        local_launcher.unlink(missing_ok=True)

    payload = "\n".join([
        "set -u",
        f"cd {base} || exit 1",
        f"bash {remote_launcher}",
    ])
    encoded = base64.b64encode(payload.encode()).decode()
    known = {j.get("jobId") for j in client.all_jobs(page_size=80)}
    submit = (
        f"cd {base} && cbatch -p TRAIN -A buu_open -q {api.queue} -N 1 "
        f"--gres gpu:{api.gpus} -t {api.minutes} "
        f"-J cx-{api.name} -o {base}/{api.name}.out -e {base}/{api.name}.err "
        f"--wrap 'echo {encoded} | base64 -d | bash'"
    )
    label = api.node or "auto"
    print(f"submitting {api.name} (node={label}) "
          f"(gpus={api.gpus}, minutes={api.minutes}, sandbox={api.sandbox})", flush=True)
    print("  " + client.run(submit, timeout=120).strip()[:400], flush=True)

    for _ in range(10):
        time.sleep(4)
        fresh = [j for j in client.all_jobs(page_size=80) if j.get("jobId") not in known]
        if fresh:
            job = max(fresh, key=lambda j: j.get("jobId"))
            print(f"  job {job.get('jobId')} state={job.get('state')} "
                  f"name={job.get('name')}", flush=True)
            return 0
    print("  could not identify the new job id（作业可能仍在排队）", flush=True)
    return 1


def cli() -> argparse.ArgumentParser:
    """命令行参数。"""
    p = argparse.ArgumentParser(description="提交常驻沙箱作业（钉节点、不等待）")
    p.add_argument("--name", required=True, help="作业名后缀，也用作日志名")
    p.add_argument("--home", default="/data/home/2024240384035", help="远端 home")
    p.add_argument("--node", default="", help="钉住的节点；空表示交给调度（两个作业会自动分到 06/07）")
    p.add_argument("--gpus", type=int, default=8, help="申请卡数")
    p.add_argument("--minutes", type=int, default=1440, help="作业时长（分钟）")
    p.add_argument("--hours", type=int, default=24, help="runner 的 LIFETIME_HOURS")
    p.add_argument("--script-timeout", type=int, default=1800, help="单脚本超时（秒）")
    p.add_argument("--sandbox", default="coffin-work/base/sandbox", help="沙箱根（相对 home）")
    p.add_argument("--queue", default="high", choices=("high", "normal", "low"))
    return p


if __name__ == "__main__":
    raise SystemExit(main(cli().parse_args()))
