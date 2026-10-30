"""video-crosscheck 命令行入口。

用法：
  python run.py --video video.m4s --transcript transcript.md --outdir out
  python run.py --transcript transcript.md --frames-dir out/frames --outdir out
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import vx


def main() -> int:
    ap = argparse.ArgumentParser(description="画面 OCR 与配音转写交叉校验")
    ap.add_argument("--video", help="本地视频文件（与 --frames-dir 二选一）")
    ap.add_argument("--frames-dir", help="已有抽帧目录（跳过抽帧）")
    ap.add_argument("--transcript", required=True, help="带时间轴的转写稿 markdown")
    ap.add_argument("--outdir", default="crosscheck_out", help="输出目录")
    ap.add_argument("--interval", type=int, default=2, help="抽帧间隔秒数（默认 2）")
    ap.add_argument("--report", default=None, help="报告输出路径")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    transcript = pathlib.Path(args.transcript)
    if not transcript.is_file():
        print("找不到转写稿：%s" % transcript, file=sys.stderr)
        return 2

    print("[1/4] 准备帧", flush=True)
    if args.frames_dir:
        frames_dir = pathlib.Path(args.frames_dir)
        files = sorted(frames_dir.glob("f_*.png"))
        if not files:
            print("帧目录里没有 f_*.png：%s" % frames_dir, file=sys.stderr)
            return 2
    elif args.video:
        frames_dir = outdir / "frames"
        files = vx.extract_frames(pathlib.Path(args.video), frames_dir, args.interval)
    else:
        print("需要 --video 或 --frames-dir", file=sys.stderr)
        return 2
    print("      帧数：%d" % len(files), flush=True)

    print("[2/4] OCR（首次较慢，结果会缓存）", flush=True)
    records = vx.ocr_frames(files, args.interval, cache=outdir / "ocr_frames.json")

    print("[3/4] 聚合成幻灯片与字幕轨", flush=True)
    slides = vx.cluster_slides(records)
    subs = vx.build_subtitles(records)
    asr = vx.parse_transcript(transcript)
    print("      幻灯片 %d 页 / 字幕 %d 行 / ASR %d 段"
          % (len(slides), len(subs), len(asr)), flush=True)

    print("[4/4] 交叉校验", flush=True)
    suspects, screen_text, gaps = vx.crosscheck(slides, subs, asr)
    md = vx.report(slides, subs, asr, suspects, screen_text, gaps,
                   {"frames": len(files)})

    out = pathlib.Path(args.report) if args.report else outdir / "crosscheck_report.md"
    out.write_text(md, encoding="utf-8")
    print("\n报告：%s" % out)
    print("疑似错字 %d 条｜配音未覆盖 %d 条｜画面其他文本 %d 条"
          % (len(suspects), len(gaps), len(screen_text)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
