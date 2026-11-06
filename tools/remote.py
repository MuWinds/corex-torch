"""corex-torch 的远端通道：登录节点执行 + 沙箱队列投递与收结果。

为什么要这层：这套调试循环里每轮都要做三件事——把探针脚本发到队列、等它跑完、把日志读回来。
裸用 ``python -m scow`` 每件都要一条进程启动（约 0.4 s 解释器 + 一次 SCOW 往返），一轮
三轮起步；本文件把这三件事压成一条命令，并提供一次上传多个脚本的批量投递（队列是串行的，
批量投递比逐个等待省掉 N-1 次往返）。

Module: ``corex-torch/tools``。在 Windows 侧运行，经 ``platform/scow`` 客户端下发，
不进入远端执行环境。

Constraints:
    * 上传的脚本正文里不能出现字面量 ``#!/bin/bash``：SCOW 网关会把含该串的文件整份过滤成
      0 字节并返回成功（``platform/scow/README.md`` §5.2）。脚本一律以 ``set -u`` 开头。
    * 队列文件名就是日志名与完成标记名，重名会互相覆盖；重跑同一个名字前先 ``sbx-reset``。
    * 远端沙箱根目录写死成 ``coffin-work/base/sandbox``，与 ``corex-triton/tools/sandbox.sh``
      的 ``SANDBOX=`` 一致；改一处就要改两处。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Iterable

#: 仓库根的上一级是 coffin-repo，platform/scow 在那里。
_PLATFORM = pathlib.Path(__file__).resolve().parents[2] / "platform"
sys.path.insert(0, str(_PLATFORM))

from scow import ScowClient  # noqa: E402

#: 远端沙箱根目录，与沙箱里 runner 的 SANDBOX= 一致。
SANDBOX = "coffin-work/base/sandbox"


def _remote(home: str, *parts: str) -> str:
    return "/".join([home, SANDBOX, *parts])


def _stage_bytes(script: pathlib.Path, env: Iterable[str]) -> pathlib.Path:
    """把脚本归一成 LF 并按需在前面加 export，返回可上传的本地文件。

    队列脚本在远端由 ``bash`` 执行，本机环境变量传不过去；把参数写成脚本头部的
    ``export`` 比在远端命令行里拼一长串引号安全。

    Args:
        script: 本地脚本路径。
        env: ``K=V`` 形式的 export 列表，可为空。

    Returns:
        实际要上传的本地文件路径。需要改写时写到同目录的 ``<名字>.env.sh`` /
        ``<名字>.lf.sh``，不改动原文件（原文件还要能重放出旧结论）。
    """
    raw = script.read_bytes()
    if b"\r\n" in raw:
        raw = raw.replace(b"\r\n", b"\n")
    env_list = list(env)
    if env_list:
        prefix = "".join(f"export {kv}\n" for kv in env_list).encode()
        staged = script.with_name(script.name + ".env.sh")
        staged.write_bytes(prefix + raw)
        return staged
    if b"\r\n" in script.read_bytes():
        staged = script.with_name(script.name + ".lf.sh")
        staged.write_bytes(raw)
        return staged
    return script


def cmd_sh(args: argparse.Namespace) -> int:
    """在登录节点执行一条（可复合的）命令。"""
    client = ScowClient()
    try:
        out, code = client.run_full(args.command, timeout=args.timeout)
        print(out)
        return 0 if code == 0 else 1
    finally:
        client.close()


def cmd_put(args: argparse.Namespace) -> int:
    """上传单个文件到远端任意路径（自动建父目录）。"""
    client = ScowClient()
    try:
        client.upload(pathlib.Path(args.local), args.remote)
        print(f"put {args.local} -> {args.remote}")
        return 0
    finally:
        client.close()


def cmd_get(args: argparse.Namespace) -> int:
    """下载远端单个文件。"""
    client = ScowClient()
    try:
        client.download(args.remote, pathlib.Path(args.local))
        print(f"get {args.remote} -> {args.local}")
        return 0
    finally:
        client.close()


def cmd_pushdir(args: argparse.Namespace) -> int:
    """把一个本地目录树打进 tar 上传，远端解开。迭代 ``src/`` 时用它。

    为什么不用逐个 ``put``：一次调试动辄要传 3–8 个源文件，逐个上传既是 N 次往返，
    又要自己保证父目录存在。tar 还顺手绕开了网关对含 ``#!/bin/bash`` 的文本做清零的坑
    （二进制载荷不受影响，见 ``platform/scow/README.md`` §5.1）。
    """
    import io
    import tarfile

    src = pathlib.Path(args.local)
    if not src.is_dir():
        print(f"不是目录：{src}", file=sys.stderr)
        return 2
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(src.rglob("*")):
            if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(src)))
    payload = buf.getvalue()
    client = ScowClient()
    try:
        home = client.config.home
        remote_tar = f"{home}/.tmp_corex_torch_push.tgz"
        local_tar = pathlib.Path(__file__).resolve().parent / ".push.tmp.tgz"
        local_tar.write_bytes(payload)
        try:
            client.upload(local_tar, remote_tar)
        finally:
            local_tar.unlink(missing_ok=True)
        cmd = (f"mkdir -p '{args.remote}' && tar xzf '{remote_tar}' -C '{args.remote}' "
               f"&& rm -f '{remote_tar}' && ls -1 '{args.remote}' | head -20")
        out, code = client.run_full(cmd)
        print(out)
        print(f"pushdir {src} -> {args.remote} ({len(payload)} B) rc={code}")
        return 0 if code == 0 else 1
    finally:
        client.close()


def cmd_status(args: argparse.Namespace) -> int:
    """一次看全：沙箱心跳、健康卡、队列、最近跑完的脚本。"""
    client = ScowClient()
    try:
        home = client.config.home
        for label, name in (("心跳", "logs/heartbeat.txt"), ("环境", "logs/env.txt")):
            try:
                print(f"=== {label} ===")
                print(client.read_text(_remote(home, name)).strip())
            except Exception as exc:                    # noqa: BLE001
                print(f"({label}读不到：{exc})")
        print("=== 队列 ===")
        try:
            entries = client.list_dir(_remote(home, "queue"))
            print("\n".join("  " + str(e.get("name")) for e in entries) or "  (空)")
        except Exception as exc:                        # noqa: BLE001
            print(f"  (队列读不到：{exc})")
        print("=== runner 尾部 ===")
        try:
            log = client.read_text(_remote(home, "logs/runner.log"))
            print("\n".join(log.strip().splitlines()[-6:]))
        except Exception as exc:                        # noqa: BLE001
            print(f"  (runner.log 读不到：{exc})")
        return 0
    finally:
        client.close()


def _submit(client: ScowClient, script: pathlib.Path, name: str | None,
            env: Iterable[str]) -> str:
    """投一个脚本进队列，返回队列名。不等待。"""
    home = client.config.home
    queue_name = name or script.stem
    rem = _remote(home, "queue", f"{queue_name}.sh")
    client.upload(_stage_bytes(script, env), rem)
    return queue_name


def _reset(client: ScowClient, names: Iterable[str]) -> None:
    """删掉完成标记，让同名脚本可以被重跑。"""
    home = client.config.home
    for name in names:
        for sub in ("done", "logs"):
            suffix = "rc" if sub == "done" else "out"
            try:
                client.remove(_remote(home, sub, f"{name}.{suffix}"))
            except Exception:                           # noqa: BLE001 - 不存在就是已删除
                pass
        for extra in ("running.sh", "ran.sh"):
            try:
                client.remove(_remote(home, "done", f"{name}.{extra}"))
            except Exception:                           # noqa: BLE001
                pass


def _wait(client: ScowClient, names: list[str], timeout: float, poll: float) -> dict[str, int | None]:
    """轮询 done/<name>.rc，直到全部有结论或超时。"""
    home = client.config.home
    pending = set(names)
    rcs: dict[str, int | None] = {}
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        for name in sorted(pending):
            try:
                text = client.read_text(_remote(home, "done", f"{name}.rc")).strip()
            except Exception:                           # noqa: BLE001 - 未完成时没有该文件
                continue
            if text:
                rcs[name] = int(text)
                pending.discard(name)
        if pending:
            time.sleep(poll)
    for name in pending:
        rcs[name] = None
    return rcs


def _print_log(client: ScowClient, name: str, tail: int) -> None:
    home = client.config.home
    try:
        out = client.read_text(_remote(home, "logs", f"{name}.out"))
    except Exception as exc:                            # noqa: BLE001
        print(f"--- {name}: 日志读取失败 {exc}")
        return
    lines = out.splitlines()
    if tail and len(lines) > tail:
        print(f"--- {name}: 只显示最后 {tail} 行（共 {len(lines)} 行）")
        lines = lines[-tail:]
    print("\n".join(lines))


def cmd_sbx_push(args: argparse.Namespace) -> int:
    """批量投递脚本，不等结果。队列串行执行，按文件名字典序。"""
    client = ScowClient()
    try:
        for path in args.scripts:
            script = pathlib.Path(path)
            name = pathlib.Path(path).stem
            print(f"queued: {_submit(client, script, args.name if len(args.scripts) == 1 else None, args.env)}")
        return 0
    finally:
        client.close()


def cmd_sbx_run(args: argparse.Namespace) -> int:
    """投一个脚本、等它跑完、把日志打回来。最常用。"""
    client = ScowClient()
    try:
        name = _submit(client, pathlib.Path(args.script), args.name, args.env)
        print(f"queued: {name}")
        rcs = _wait(client, [name], args.timeout, args.poll)
        rc = rcs[name]
        print(f"rc={rc}")
        _print_log(client, name, args.tail)
        return 0 if rc == 0 else 1
    finally:
        client.close()


def cmd_sbx_wait(args: argparse.Namespace) -> int:
    """等一批已在队列里的脚本跑完，逐个打日志。"""
    client = ScowClient()
    try:
        rcs = _wait(client, list(args.names), args.timeout, args.poll)
        for name in args.names:
            rc = rcs.get(name)
            print(f"=== {name} rc={rc} ===")
            _print_log(client, name, args.tail)
        return 0 if all(v == 0 for v in rcs.values()) else 1
    finally:
        client.close()


def cmd_sbx_log(args: argparse.Namespace) -> int:
    """只读一个已有脚本的日志。"""
    client = ScowClient()
    try:
        _print_log(client, args.name, args.tail)
        return 0
    finally:
        client.close()


def cmd_sbx_reset(args: argparse.Namespace) -> int:
    """清掉完成标记与日志，好让同名脚本重跑。"""
    client = ScowClient()
    try:
        _reset(client, list(args.names))
        print("reset:", " ".join(args.names))
        return 0
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="corex-torch 远端通道")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sh", help="登录节点执行")
    s.add_argument("command")
    s.add_argument("--timeout", type=float, default=120.0)
    s.set_defaults(func=cmd_sh)

    s = sub.add_parser("put", help="上传单个文件")
    s.add_argument("local")
    s.add_argument("remote")
    s.set_defaults(func=cmd_put)

    s = sub.add_parser("get", help="下载单个文件")
    s.add_argument("remote")
    s.add_argument("local")
    s.set_defaults(func=cmd_get)

    s = sub.add_parser("pushdir", help="把目录打成 tar 上传并解开")
    s.add_argument("local")
    s.add_argument("remote")
    s.set_defaults(func=cmd_pushdir)

    s = sub.add_parser("status", help="沙箱心跳/队列/runner")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("sbx-push", help="批量投递，不等")
    s.add_argument("scripts", nargs="+")
    s.add_argument("--name", default=None)
    s.add_argument("--env", action="append", default=[], metavar="K=V")
    s.set_defaults(func=cmd_sbx_push)

    s = sub.add_parser("sbx-run", help="投递并等待")
    s.add_argument("script")
    s.add_argument("--name", default=None)
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--poll", type=float, default=4.0)
    s.add_argument("--tail", type=int, default=0)
    s.add_argument("--env", action="append", default=[], metavar="K=V")
    s.set_defaults(func=cmd_sbx_run)

    s = sub.add_parser("sbx-wait", help="等一批已投递的脚本")
    s.add_argument("names", nargs="+")
    s.add_argument("--timeout", type=float, default=1800.0)
    s.add_argument("--poll", type=float, default=4.0)
    s.add_argument("--tail", type=int, default=0)
    s.set_defaults(func=cmd_sbx_wait)

    s = sub.add_parser("sbx-log", help="读已有日志")
    s.add_argument("name")
    s.add_argument("--tail", type=int, default=0)
    s.set_defaults(func=cmd_sbx_log)

    s = sub.add_parser("sbx-reset", help="清完成标记，允许重跑")
    s.add_argument("names", nargs="+")
    s.set_defaults(func=cmd_sbx_reset)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
