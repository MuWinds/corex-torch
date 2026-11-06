"""纯几何helper：算 strides、判连续、推定 view 之后的 strides。**不碰设备、不碰 torch**。

为什么单独一个文件：这批算子（view / as_strided / cat / stack）的错法几乎都是
「形状对了、stride 错了」，而 stride 是可以在没有卡、没有 torch 的机器上算清楚并单测的。
把这份数学从注册代码里拆出来，就能在登录节点直接跑断言（登录节点不能 import torch），
真机上只剩「按算好的 strides 去建视图」这一步。

Module: ``corex_torch``。

Constraints:
    * 不 import torch：保持「能在最裸的环境里验证」这一属性（与 ``_runtime`` 同一条纪律）。
    * 所有函数只吃/回 Python int 列表，不做任何设备访问。
"""

from __future__ import annotations

from typing import Sequence


def prod(values: Sequence[int]) -> int:
    """连乘。空序列返回 1（0 维标量的 numel 就是 1）。

    Args:
        values: 各维长度。

    Returns:
        元素总数。
    """
    total = 1
    for value in values:
        total *= int(value)
    return total


def contiguous_strides(size: Sequence[int]) -> list[int]:
    """行主序（C 序）strides。

    Args:
        size: 各维长度。

    Returns:
        与 ``size`` 等长的 strides。
    """
    strides: list[int] = []
    acc = 1
    for dim in reversed([int(s) for s in size]):
        strides.append(acc)
        acc *= dim
    return list(reversed(strides))


def is_contiguous(size: Sequence[int], stride: Sequence[int]) -> bool:
    """判断给定 size/stride 是不是行主序连续。

    为什么不用 ``tensor.is_contiguous()``：算 view 的目标 strides 时源张量可能还没建出来，
    这里要的是「只看数字就能判」的版本。

    Args:
        size: 各维长度。
        stride: 各维步长。

    Returns:
        连续为 True。size 里有 0 时按「空张量」处理，返回 True。
    """
    expected = 1
    for dim in range(len(size) - 1, -1, -1):
        if int(size[dim]) == 0:
            return True
        if int(size[dim]) != 1 and int(stride[dim]) != expected:
            return False
        expected *= int(size[dim])
    return True


def storage_span(size: Sequence[int], stride: Sequence[int]) -> int:
    """这套 size/stride 需要多少个元素的存储（= 最大元素下标 + 1）。

    为什么要它：`as_strided` 的越界检查与 `empty_strided` 的分配量都靠这个数，而它**不是**
    numel（有空洞/转置时更大）。曾经这里写成「取各维最大跨度」，多维张量会少算
    （``docs/03-TORCH28-API.md`` §10.5 有复盘）。

    Args:
        size: 各维长度。
        stride: 各维步长。

    Returns:
        需要的元素个数；空张量返回 0。
    """
    if any(int(s) <= 0 for s in size):
        return 0
    elements = 0
    for dim, length in enumerate(size):
        elements += int(stride[dim]) * (int(length) - 1)
    return elements + 1


def infer_view_size(size: Sequence[int], numel: int) -> list[int]:
    """把 ``view`` 目标形状里的 ``-1`` 推出来。

    为什么要有：``embedding`` 的 composite 第一步就是 ``index.view(-1)``（实测报错
    ``view 的 numel 不一致：4 -> -1``）。不处理 -1 的话它一步都走不了。

    Args:
        size: 目标形状，最多一个 -1。
        numel: 源张量元素总数。

    Returns:
        推好的形状。

    Raises:
        ValueError: -1 超过一个，或推出非整数/为负，或总元素数不一致。
    """
    out = [int(s) for s in size]
    if out.count(-1) > 1:
        raise ValueError(f"view 的形状里只能有一个 -1：{out}")
    known = prod([s for s in out if s != -1])
    if -1 in out:
        if known == 0 or numel % known != 0:
            raise ValueError(f"view 的 -1 推不出来：numel={numel} 已知部分={known}")
        out[out.index(-1)] = numel // known
    if prod(out) != numel:
        raise ValueError(f"view 的 numel 不一致：{numel} -> {prod(out)}")
    return out


def view_strides(size: Sequence[int], stride: Sequence[int],
                 new_size: Sequence[int]) -> list[int]:
    """算 ``Tensor.view(new_size)`` 之后的 strides。

    只处理**连续源张量**这一种情况（覆盖实际调用里的绝大多数），不连续时抛
    ``ValueError`` 交给调用方降级——为什么不做完整的 collapse-split：那套算法在 size-1 维上
    产生的 stride 是「任意合法值」，与 CPU 参考实现不一定逐值相同，硬比 stride 会得到
    假失败；而 ``reshape``（composite）本来就会在不可视图时退化成 copy+view，语义更稳。

    Args:
        size: 源形状。
        stride: 源 strides。
        new_size: 目标形状（可以含一个 -1）。

    Returns:
        目标 strides。

    Raises:
        ValueError: numel 不等，或源张量不连续（需要先 contiguous）。
    """
    resolved = infer_view_size(new_size, prod(size))
    if not is_contiguous(size, stride):
        raise ValueError("源张量不连续，view 需要先 contiguous()（或用 reshape 让它自己 copy）")
    return contiguous_strides(resolved)


def check_cat_inputs(shapes: Sequence[Sequence[int]], dim: int) -> list[int]:
    """校验 cat 的输入形状，返回拼接后的形状。

    Args:
        shapes: 每个输入的形状。
        dim: 拼接维。

    Returns:
        拼接后的形状。

    Raises:
        ValueError: 维数不一致，或非拼接维长度不同。
    """
    if not shapes:
        raise ValueError("cat 至少需要一个输入")
    rank = len(shapes[0])
    for shape in shapes:
        if len(shape) != rank:
            raise ValueError(f"cat 的输入维数不一致：{len(shape)} vs {rank}")
    out = [int(x) for x in shapes[0]]
    for shape in shapes[1:]:
        for axis, (a, b) in enumerate(zip(out, shape)):
            if axis != dim and int(a) != int(b):
                raise ValueError(f"cat 在第 {axis} 维长度不一致：{a} vs {b}")
    out[dim] = sum(int(shape[dim]) for shape in shapes)
    return out
