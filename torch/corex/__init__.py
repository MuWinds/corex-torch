"""corex 设备后端：把天垓100 通过 ``torch.device("corex")`` 交给上游 PyTorch 2.8。

这个包分三层，各自可以单独验证：

* ``_runtime``：ctypes 直连 CoreX 的 ``libcudart``——设备、显存、拷贝、流、事件。**不 import torch**。
* ``_triton``（待建）：把 ``triton_corex`` 的编译链接过来，补上它原本向厂商 torch 借的
  context / stream / 设备属性。
* ``_device``（待建）：把上面两层注册成 torch 的 ``PrivateUse1`` 设备，并实现第一批 aten 算子。

Module: ``torch.corex``。上游只有一个入口：用户 ``import torch``（fork 自带本包）或显式 ``import torch.corex`` 之后，
``torch.device("corex")`` 才存在。

Constraints:
    * **import 本包时不许碰设备**：文档、CI、登录节点（没有 ``/usr/local/corex``）都可能
      只是 import 一下。设备只在第一次真正分配时初始化。
    * **不提供任何 CUDA graph 能力**：KMD 没有图通道，用 CUDA Graph API 会挂卡
      （``docs/05-DRIVER-CONSTRAINTS.md`` R2）。
    * 本模块是**唯一**允许 ``import torch`` 的地方之一（注册设备时）；``_runtime`` 永远不 import。
"""

from __future__ import annotations

from . import _runtime

__all__ = [
    "is_available",
    "device_count",
    "current_device",
    "set_device",
    "synchronize",
    "properties",
    "Stream",
    "Event",
    "register",
]

#: 设备名。与 ``torch.utils.rename_privateuse1_backend`` 一致，也是用户写的那个字符串。
DEVICE_NAME = "corex"

#: 设备后端版本，写进 ``torch.corex`` 模块的属性，便于在报错信息里对账。
__version__ = "0.1.0"

#: ``Stream`` / ``Event`` 直接复用运行时的实现（它们本来就是句柄 + 同步语义）。
Stream = _runtime.Stream
Event = _runtime.Event


def is_available() -> bool:
    """CoreX 运行时是否可用。不抛异常，导入期也不会触发设备初始化。"""
    return _runtime.is_available()


def device_count() -> int:
    """设备数。"""
    return _runtime.device_count()


def current_device() -> int:
    """当前设备号。"""
    return _runtime.current_device()


def set_device(index: int) -> None:
    """切换当前设备。"""
    _runtime.set_device(index)


def synchronize() -> None:
    """等当前设备上所有工作完成。"""
    _runtime.synchronize()


def properties(index: int = 0) -> dict:
    """设备属性字典（名字、warp 宽度、shared 上限、显存）。"""
    return _runtime.properties(index)


def register() -> None:
    """把 ``corex`` 注册成 torch 的设备。

    这一步要等 ``_device`` 写完才算完整（分配器、设备属性、aten 分发键）；现在是个占位，
    存在的意义是让调用方有稳定的入口：先 ``import torch; torch.corex.register()``，
    内部怎么实现随阶段推进换，调用点不改。

    Raises:
        RuntimeError: torch 版本不满足 2.8 这一代（本仓库只针对 2.8 的 PrivateUse1 接口写）。
    """
    import torch

    major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if (major, minor) != (2, 8):
        raise RuntimeError(
            f"corex 后端只针对 torch 2.8 写（实测接口见 docs/03-TORCH28-API.md），当前是 {torch.__version__}"
        )
    from . import _device  # noqa: F401 - 导入即完成注册（分配器 + aten 分发）

    _device.register()
