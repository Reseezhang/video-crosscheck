"""视频双证据链校验：画面 OCR 与配音转写交叉比对。

设计要点（都是踩过坑换来的，改动前先读）：
1. 帧聚类用集合 Jaccard，不用序列相似度 —— OCR 同一页的行序不稳定。
2. 字幕带不能用固定 y 阈值 —— 视频里会出现全屏截图/游戏画面，必须叠加
   「居中 + 长度 + 中文占比」三重过滤。
3. 比对必须用最长公共子串做包含匹配 —— 用 difflib.ratio 拿一句字幕去比
   整段 ASR，假阳性会超过 60%（已在真实样本上验证过）。
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from difflib import SequenceMatcher

PUNCT_RE = re.compile(
    r"[\s，。、；：？！“”‘’（）《》〈〉—…·,.;:?!\"'()\[\]<>|/\\~`@#$%^&*+=_-]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
TS_RE = re.compile(r"\[(\d+):(\d+)-(\d+):(\d+)\]\s*(.+)")

# 相对阈值（按画面高度比例），不写死像素
WATERMARK_Y = 0.13       # 顶部水印/平台角标
SUBTITLE_Y = 0.80        # 字幕带起点
SLIDE_END_Y = 0.78       # 幻灯片正文下界
SUB_MAX_LEN = 42         # 单行字幕最长字数
SUB_CJK_RATIO = 0.5      # 字幕以中文为主

OCR_MIN_SCORE = 0.5
SLIDE_JACCARD = 0.5      # 相邻帧归为同一张幻灯片的下限
SLIDE_MIN_FRAME_RATIO = 0.4   # 一行文字要在本页多数帧出现才算数
GAP_RATIO = 0.62         # 幻灯片文字在 ASR 中的最低覆盖率，低于即视为"配音没讲"
SUSPECT_LO = 0.80        # 字幕与配音"同一句"的吻合度下界
SUSPECT_HI = 0.98        # 高于此值视为一致，不再打扰人看


def norm(s: str) -> str:
    return PUNCT_RE.sub("", s).lower()


def ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def image_height(path: pathlib.Path) -> int:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.height
    except Exception:
        return 480


BILI_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


def parse_bvid(text: str) -> str | None:
    """从分享文本 / 链接 / 裸 BV 号里取出 BV 号。"""
    m = re.search(r"(BV[0-9A-Za-z]{10})", text or "")
    return m.group(1) if m else None


def _bili_json(url: str, params: dict, headers: dict, timeout: int = 20) -> dict:
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{query}", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_bilibili_video(target: str, outdir: pathlib.Path, page: int = 1) -> pathlib.Path:
    """从 B 站链接/BV 号直接取视频流。

    注意：未登录状态下 B 站只给 480p（`accept_quality` 会列出 1080P，但 dash 里没有）。
    480p 对 OCR 通常够用，前提是原视频字号够大。
    """
    bvid = parse_bvid(target) or target.strip()
    headers = {"User-Agent": BILI_UA,
               "Referer": f"https://www.bilibili.com/video/{bvid}/"}

    view = _bili_json("https://api.bilibili.com/x/web-interface/view",
                      {"bvid": bvid}, headers)
    if view.get("code") != 0:
        raise RuntimeError("取视频信息失败：%s" % view.get("message"))
    data = view["data"]
    pages = data.get("pages") or [{"cid": data["cid"]}]
    idx = max(1, min(page, len(pages))) - 1
    cid = pages[idx]["cid"]

    play = _bili_json("https://api.bilibili.com/x/player/playurl",
                      {"bvid": bvid, "cid": cid, "fnval": 16, "qn": 127}, headers)
    if play.get("code") != 0:
        raise RuntimeError("取播放地址失败：%s" % play.get("message"))
    videos = (play.get("data") or {}).get("dash", {}).get("video") or []
    if not videos:
        raise RuntimeError("没有可用的视频流（可能是付费/会员内容）")
    best = max(videos, key=lambda v: v.get("width", 0) * v.get("height", 0))

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("%s_p%d.m4s" % (bvid, idx + 1))
    if not out.is_file() or out.stat().st_size < 1024:
        req = urllib.request.Request(best["baseUrl"], headers=headers)
        with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
    print("  视频：%s（%dx%d，%.1f MB）"
          % (out.name, best.get("width", 0), best.get("height", 0),
             out.stat().st_size / 1048576), flush=True)
    return out


def extract_frames(video: pathlib.Path, outdir: pathlib.Path, interval: int = 2):
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("f_*.png"):
        old.unlink()
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-vf", f"fps=1/{interval}", "-q:v", "2", "-y",
           str(outdir / "f_%04d.png")]
    subprocess.run(cmd, check=True)
    return sorted(outdir.glob("f_*.png"))


def ocr_frames(files, interval: int = 2, cache: pathlib.Path | None = None):
    if cache and cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    h = image_height(files[0])
    sub_y = h * SUBTITLE_Y
    mark_y = h * WATERMARK_Y

    records = []
    for i, fp in enumerate(files, start=1):
        ts = (i - 1) * interval
        result, _ = engine(str(fp))
        content, subtitle = [], []
        for item in (result or []):
            box, text, score = item[0], item[1], float(item[2])
            if score < OCR_MIN_SCORE:
                continue
            ys = [float(p[1]) for p in box]
            xs = [float(p[0]) for p in box]
            y, hgt = min(ys), max(ys) - min(ys)
            cy = y + hgt / 2
            if y < mark_y:
                continue
            rec = {"text": text, "score": round(score, 3), "y": int(y),
                   "x": int(min(xs)), "w": int(max(xs) - min(xs))}
            if cy >= sub_y:
                subtitle.append(rec)
            elif cy <= h * SLIDE_END_Y:
                content.append(rec)
        records.append({"i": i, "ts": ts,
                        "content": [r["text"] for r in content],
                        "subtitle_detail": subtitle})
        if i % 25 == 0 or i == len(files):
            print(f"  OCR {i}/{len(files)}", flush=True)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def cluster_slides(records):
    """把逐帧文字按集合相似度聚成幻灯片页。"""
    segments, cur = [], None
    for rec in records:
        lines = [l for l in rec["content"] if norm(l)]
        keys = {norm(l) for l in lines}
        if cur is None or _jaccard(keys, set(cur["lines"])) < SLIDE_JACCARD:
            cur = {"start": rec["ts"], "end": rec["ts"], "lines": {}, "frames": 0}
            segments.append(cur)
        cur["end"] = rec["ts"]
        cur["frames"] += 1
        for l in lines:
            cur["lines"][l] = cur["lines"].get(l, 0) + 1

    slides = []
    for seg in segments:
        keep = [l for l, c in seg["lines"].items()
                if c >= max(2, seg["frames"] * SLIDE_MIN_FRAME_RATIO)]
        if keep:
            keep.sort(key=lambda x: -seg["lines"][x])
            slides.append({"start": seg["start"], "end": seg["end"],
                           "frames": seg["frames"], "lines": keep})
    return slides


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_subtitles(records):
    """重建字幕轨：位于底部 + 够短 + 以中文为主，避免把游戏文本当成字幕。"""
    subs, last = [], None
    for rec in records:
        picked = []
        for d in rec["subtitle_detail"]:
            t = d["text"]
            if len(t) > SUB_MAX_LEN:
                continue
            cjk = len(CJK_RE.findall(t))
            if cjk < len(t) * SUB_CJK_RATIO:
                continue
            picked.append(t)
        text = "".join(picked)
        key = norm(text)
        if key and key != last:
            subs.append({"ts": rec["ts"], "text": text})
            last = key
    return subs


def parse_transcript(path: pathlib.Path):
    asr = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = TS_RE.match(line.strip())
        if m:
            asr.append({
                "start": int(m.group(1)) * 60 + int(m.group(2)),
                "end": int(m.group(3)) * 60 + int(m.group(4)),
                "text": m.group(5).strip(),
            })
    return asr


def _index(asr):
    full, owner = "", []
    for i, a in enumerate(asr):
        t = norm(a["text"])
        full += t
        owner.extend([i] * len(t))
    return full, owner


def _probe(full, owner, asr, query: str):
    """最长公共子串占比 + 命中段落。"""
    if not query:
        return 0.0, None
    m = SequenceMatcher(None, full, query, autojunk=False).find_longest_match(
        0, len(full), 0, len(query))
    if m.size == 0:
        return 0.0, None
    seg = asr[owner[m.a]] if m.a < len(owner) else None
    return m.size / len(query), seg


def _best_window(seg_norm: str, q: str):
    """在 ASR 段落里滑窗找与 q 最贴合的一段，返回 (吻合度, 窗口文本)。

    只比整段会因长度差异失真；只比 LCS 又不给窗口，无法做字符级 diff。
    """
    n = len(q)
    if n == 0 or not seg_norm:
        return 0.0, ""
    best_r, best_w = 0.0, ""
    for i in range(0, max(1, len(seg_norm) - n + 1)):
        w = seg_norm[i:i + n]
        s = SequenceMatcher(None, w, q, autojunk=False).find_longest_match(
            0, len(w), 0, len(q)).size
        r = s / n
        if r > best_r:
            best_r, best_w = r, w
            if r == 1.0:
                break
    return best_r, best_w


def _char_diff(asr_part: str, ocr_part: str) -> str:
    """字符级差异，直接给出『配音 → 画面』的对照片段。"""
    sm = SequenceMatcher(None, asr_part, ocr_part, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        parts.append("%s → %s" % (asr_part[i1:i2] or "∅", ocr_part[j1:j2] or "∅"))
    return "；".join(parts)


def crosscheck(slides, subs, asr):
    """三级分类：几乎一致 / 疑似错字（局部差异）/ 画面其他文本。"""
    full, owner = _index(asr)

    suspects, screen_text = [], []
    for s in subs:
        q = norm(s["text"])
        if len(q) < 5:
            continue
        m = SequenceMatcher(None, full, q, autojunk=False).find_longest_match(
            0, len(full), 0, len(q))
        if m.size == 0:
            screen_text.append({"ts": s["ts"], "text": s["text"]})
            continue
        seg = asr[owner[m.a]] if m.a < len(owner) else None
        if seg is None:
            continue
        r, window = _best_window(norm(seg["text"]), q)
        if r >= SUSPECT_HI:
            continue                      # 与配音一致，无需人工看
        if r >= SUSPECT_LO:
            suspects.append({"ts": s["ts"], "ratio": round(r, 2),
                             "ocr": s["text"], "asr": seg["text"],
                             "diff": _char_diff(window, q)})
        else:
            screen_text.append({"ts": s["ts"], "text": s["text"]})

    gaps = []
    for sl in slides:
        for line in sl["lines"]:
            q = norm(line)
            if len(q) < 3:
                continue
            m = SequenceMatcher(None, full, q, autojunk=False).find_longest_match(
                0, len(full), 0, len(q))
            r = m.size / len(q)
            if r < GAP_RATIO:
                gaps.append({"start": sl["start"], "text": line, "ratio": round(r, 2)})
    return suspects, screen_text, gaps


def _mmss(s: int) -> str:
    return "%02d:%02d" % (s // 60, s % 60)


def report(slides, subs, asr, suspects, screen_text, gaps, meta: dict) -> str:
    out = ["# 双证据链校验报告", ""]
    out.append("| 项目 | 数量 |")
    out.append("|---|---|")
    out.append("| OCR 帧 | %d |" % meta.get("frames", 0))
    out.append("| 幻灯片页 | %d |" % len(slides))
    out.append("| 字幕行 | %d |" % len(subs))
    out.append("| ASR 段落 | %d |" % len(asr))
    out.append("| **疑似错字** | **%d** |" % len(suspects))
    out.append("| 配音未覆盖 | %d |" % len(gaps))
    out.append("| 画面其他文本（非字幕） | %d |" % len(screen_text))
    out.append("")

    out.append("## 一、疑似 ASR 错字（优先看这里）")
    out.append("")
    out.append("> 画面字幕逐字打在屏上，通常比语音识别可靠。下面是字符级差异，"
               "`配音 → 画面` 的写法即修正依据。")
    out.append("")
    for d in suspects:
        out.append("- `t=%ds` 吻合度 %.2f" % (d["ts"], d["ratio"]))
        out.append("  - 差异：%s" % d["diff"])
        out.append("  - 配音整句：%s" % d["asr"])
    out.append("")

    out.append("## 二、配音未覆盖的幻灯片内容")
    out.append("")
    out.append("> 这些文字只出现在画面上，配音没有念到，是纯看转写会漏掉的信息。")
    out.append("")
    by_slide = {}
    for g in gaps:
        by_slide.setdefault(g["start"], []).append(g["text"])
    for start in sorted(by_slide):
        out.append("- **%s**" % _mmss(start))
        for t in by_slide[start]:
            out.append("  - %s" % t)
    out.append("")

    out.append("## 三、画面上的其他文本（与配音无关，可忽略）")
    out.append("")
    out.append("> 多半是游戏内文本、图表标注、聊天截图。列出来是为了说明它们没被当成字幕误判。")
    out.append("")
    for s in screen_text[:30]:
        out.append("- `t=%ds` %s" % (s["ts"], s["text"]))
    if len(screen_text) > 30:
        out.append("- ……另有 %d 条" % (len(screen_text) - 30))
    out.append("")

    out.append("## 四、幻灯片时间轴")
    out.append("")
    for i, sl in enumerate(slides, 1):
        out.append("- **%s-%s** slide %02d：%s"
                   % (_mmss(sl["start"]), _mmss(sl["end"]), i,
                      " / ".join(sl["lines"][:6])))
    out.append("")
    return "\n".join(out)
