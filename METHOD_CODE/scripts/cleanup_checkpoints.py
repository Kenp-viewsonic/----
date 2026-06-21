#!/usr/bin/env python3
"""
清理 outputs 目录下的中间 checkpoint（checkpoint-{step}）。

这些中间 checkpoint 是 HuggingFace Trainer 按 epoch 保存的，从未被加载使用，
但每个 ~500MB，占用大量磁盘空间。

保留：
  - checkpoint-best/        → 被下一阶段训练加载
  - checkpoint-pre-consolidation/  → 诊断用
删除：
  - checkpoint-{step}/      → 按 epoch 的中间 checkpoint（如 checkpoint-8, checkpoint-120 等）

用法：
    python scripts/cleanup_checkpoints.py                    # 预览（dry-run）
    python scripts/cleanup_checkpoints.py --force            # 确认删除
    python scripts/cleanup_checkpoints.py --outputs ./outputs  # 指定 outputs 目录
    python scripts/cleanup_checkpoints.py --include-pre-consolidation  # 同时删除 pre-consolidation
"""

import os
import sys
import argparse
import shutil
import re
from pathlib import Path


def find_checkpoints_to_clean(outputs_root, include_pre_consolidation=False):
    """
    扫描 outputs 目录，找出所有可清理的 checkpoint 目录。

    返回 (to_delete: list[Path], total_bytes: int)
    """
    to_delete = []
    total_bytes = 0

    outputs_path = Path(outputs_root)
    if not outputs_path.is_dir():
        print(f"[ERROR] Outputs directory not found: {outputs_root}")
        return [], 0

    # 匹配 checkpoint-数字 模式（如 checkpoint-8, checkpoint-120），排除 checkpoint-best
    step_pattern = re.compile(r"^checkpoint-\d+$")
    # pre-consolidation 目录
    pre_consolidation_pattern = re.compile(r"^checkpoint-pre-consolidation$")

    for dirpath, dirnames, filenames in os.walk(outputs_path):
        current_dir = Path(dirpath)
        dir_name = current_dir.name

        should_delete = False
        reason = ""

        if step_pattern.match(dir_name):
            should_delete = True
            reason = "中间 checkpoint (按 epoch 保存)"
        elif pre_consolidation_pattern.match(dir_name) and include_pre_consolidation:
            should_delete = True
            reason = "pre-consolidation 诊断 checkpoint"

        if should_delete:
            # 计算该目录大小
            dir_size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for root, _, files in os.walk(dirpath)
                for f in files
            )
            total_bytes += dir_size
            to_delete.append((current_dir, dir_size, reason))

    return to_delete, total_bytes


def format_size(bytes_val):
    """人性化显示文件大小"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_val) < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="清理 outputs 目录下未被使用的中间 checkpoint"
    )
    parser.add_argument(
        "--outputs", type=str, default="./outputs",
        help="outputs 目录路径（默认 ./outputs）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="确认删除（否则只预览）"
    )
    parser.add_argument(
        "--include-pre-consolidation", action="store_true",
        help="同时删除 checkpoint-pre-consolidation 诊断目录"
    )
    args = parser.parse_args()

    to_delete, total_bytes = find_checkpoints_to_clean(
        args.outputs, args.include_pre_consolidation
    )

    if not to_delete:
        print("✅ 没有找到可清理的中间 checkpoint。")
        return

    print(f"📋 找到 {len(to_delete)} 个可清理的 checkpoint 目录")
    print(f"💾 预计释放空间: {format_size(total_bytes)}")
    print()

    # 按路径分组显示
    current_run = None
    for path, size, reason in sorted(to_delete):
        # 提取 run 名称用于分组
        run_part = str(path.parent.parent.parent) if "runs" in str(path) else str(path.parent)
        if run_part != current_run:
            if current_run is not None:
                print()
            current_run = run_part
            print(f"  📁 {os.path.relpath(path.parent, args.outputs)}")
        print(f"      ├── {path.name}  ({format_size(size)})  [{reason}]")
    
    print()
    print(f"{'='*60}")

    if args.force:
        print("🗑️  正在删除...")
        deleted_count = 0
        failed = []
        for path, _, reason in to_delete:
            try:
                shutil.rmtree(path)
                deleted_count += 1
            except Exception as e:
                failed.append((path, str(e)))
        
        print(f"✅ 已删除 {deleted_count}/{len(to_delete)} 个目录")
        if failed:
            print(f"❌ {len(failed)} 个删除失败:")
            for p, err in failed:
                print(f"   - {p}: {err}")
    else:
        print("ℹ️  这是预览模式（dry-run），未实际删除任何文件。")
        print(f"   执行以下命令确认删除：")
        print(f"   python scripts/cleanup_checkpoints.py --force")
        if args.include_pre_consolidation:
            print(f"   （已包含 --include-pre-consolidation）")


if __name__ == "__main__":
    main()
