import argparse
import os
import sys
from pathlib import Path

from deep_research.graph import build_graph
from deep_research.tools import slugify

REQUIRED_KEYS = ("ANTHROPIC_API_KEY", "TAVILY_API_KEY")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument("topic", help="研究问题")
    parser.add_argument("--max-iters", type=int, default=3, help="检索轮数上限（熔断）")
    parser.add_argument("--out", default="reports", help="报告输出目录")
    parser.add_argument("--thread-id", default=None, help="checkpointer 线程 id（默认按 topic 生成）")
    parser.add_argument("--sqlite", default="research.sqlite", help="checkpointer 落盘路径")
    return parser.parse_args(argv)


def check_keys() -> None:
    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        print(f"[error] 缺少环境变量: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def report_path(out_dir: str, topic: str) -> Path:
    return Path(out_dir) / f"{slugify(topic)}.md"


def main(argv=None) -> None:
    args = parse_args(argv)
    check_keys()

    thread_id = args.thread_id or slugify(args.topic)
    inputs = {"topic": args.topic, "max_iterations": args.max_iters}
    cfg = {"configurable": {"thread_id": thread_id}}

    from langgraph.checkpoint.sqlite import SqliteSaver
    with SqliteSaver.from_conn_string(args.sqlite) as saver:
        app = build_graph().compile(checkpointer=saver)
        for event in app.stream(inputs, cfg, stream_mode="updates"):
            for node in event:
                print(f"== {node} 完成 ==")
        report = app.get_state(cfg).values.get("report", "")

    if not report.strip():
        print("[error] 资料不足，未生成报告", file=sys.stderr)
        sys.exit(1)

    out_path = report_path(args.out, args.topic)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"报告已写入 {out_path}")


if __name__ == "__main__":
    main()
