# -*- coding: utf-8 -*-
"""B 题论文流程图（solve 工程方法版）。
统一分支模式：菱形判断后，两个 300-320pt 窄分支框并排放置，箭头从菱形底部两侧引出，
汇合后进入后续框。原全宽分支框在生成后立即按"文本+全宽"匹配删除。"""
import os
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

PT = 12700
FONT = "Microsoft YaHei"

C_PROC_F, C_PROC_L = RGBColor(0xE8, 0xF1, 0xFA), RGBColor(0x2E, 0x74, 0xB5)
C_DEC_F,  C_DEC_L  = RGBColor(0xFF, 0xF3, 0xD6), RGBColor(0xBF, 0x8F, 0x00)
C_TERM_F, C_TERM_L = RGBColor(0xE2, 0xEF, 0xDA), RGBColor(0x53, 0x81, 0x35)
C_IO_F,   C_IO_L   = RGBColor(0xFC, 0xEE, 0xE7), RGBColor(0xC5, 0x5A, 0x11)
C_LINE = RGBColor(0x40, 0x40, 0x40)
GREEN = RGBColor(0x53, 0x81, 0x35)
RED = RGBColor(0xC0, 0x39, 0x2B)


def _style_run(r, size, bold):
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {})
            rPr.append(e)
        e.set("typeface", FONT)


def box_h(text, size, kind):
    n = text.count("\n") + 1
    if kind == "dec":
        return max(120, n * size * 1.95 + 24)
    return max(64, n * size * 1.32 + 26)


def add_box(slide, x, y, w, text, kind="proc", size=30, bold=False, h=None):
    fill, line = {
        "proc": (C_PROC_F, C_PROC_L), "dec": (C_DEC_F, C_DEC_L),
        "term": (C_TERM_F, C_TERM_L), "io": (C_IO_F, C_IO_L),
    }[kind]
    shape = MSO_SHAPE.DIAMOND if kind == "dec" else MSO_SHAPE.ROUNDED_RECTANGLE
    if h is None:
        h = box_h(text, size, kind)
    shp = slide.shapes.add_shape(shape, Emu(int(x * PT)), Emu(int(y * PT)), Emu(int(w * PT)), Emu(int(h * PT)))
    if kind != "dec":
        try:
            shp.adjustments[0] = 0.10
        except Exception:
            pass
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line; shp.line.width = Pt(1.75)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Emu(int(6 * PT))
    tf.margin_top = tf.margin_bottom = Emu(int(2 * PT))
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 1.05
        for r in p.runs:
            _style_run(r, size, bold)
    return shp, h


def add_arrow(slide, x1, y1, x2, y2, head=True):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(x1 * PT)), Emu(int(y1 * PT)), Emu(int(x2 * PT)), Emu(int(y2 * PT)))
    c.line.color.rgb = C_LINE; c.line.width = Pt(2.0)
    c.shadow.inherit = False
    if head:
        ln = c._element.spPr.find(qn("a:ln"))
        e = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(e)
    return c


def add_label(slide, x, y, text, size=28, color=RGBColor(0x40, 0x40, 0x40), w=160):
    tb = slide.shapes.add_textbox(Emu(int(x * PT)), Emu(int(y * PT)), Emu(int(w * PT)), Emu(int((size + 12) * PT)))
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]; p.text = text
    for r in p.runs:
        _style_run(r, size, False)
        r.font.color.rgb = color
    return tb


def delete_fullwidth_box(slide, key, bw):
    """删除 vertical_flow 生成的、文本含 key 的全宽框。"""
    for sh in list(slide.shapes):
        try:
            if sh.has_text_frame and key in sh.text_frame.text and int(sh.width) == int(bw * PT):
                sh._element.getparent().remove(sh._element)
        except Exception:
            pass


def new_slide(w_pt, h_pt):
    prs = Presentation()
    prs.slide_width = Emu(int(w_pt * PT))
    prs.slide_height = Emu(int(h_pt * PT))
    prs.slides.add_slide(prs.slide_layouts[6])
    return prs


def vertical_flow(w, items, bw, gap, size=30, skip_auto=()):
    hs = [box_h(t, sz or size, k) for (t, k, sz) in items]
    total = sum(hs) + gap * (len(items) - 1)
    H = total + 2 * 34
    prs = new_slide(w, H)
    s = prs.slides[0]
    xc = (w - bw) / 2
    ys = []
    y = 34
    for (t, k, sz), h in zip(items, hs):
        add_box(s, xc, y, bw, t, k, size=sz or size, h=h)
        ys.append((y, y + h))
        y += h + gap
    for i in range(len(items) - 1):
        if i in skip_auto:
            continue
        add_arrow(s, xc + bw / 2, ys[i][1], xc + bw / 2, ys[i + 1][0])
    return prs, ys, xc, bw, hs


def branch_pair(s, cx, yd, yA, hA, textA, yB, hB, textB, dx=200, bwd=310,
                labA=("否", RED), labB=("是", GREEN), sizeA=28, sizeB=28):
    """在菱形下方放左右两个窄分支框，画分支箭头与标签；textB 为空则不画右框与右箭头。"""
    xA = cx - dx - bwd / 2
    xB = cx + dx - bwd / 2
    add_box(s, xA, yA, bwd, textA, "term", size=sizeA, h=hA)
    add_arrow(s, cx - 105, yd, cx - dx + 60, yA)
    if textB:
        add_box(s, xB, yB, bwd, textB, "term", size=sizeB, h=hB)
        add_arrow(s, cx + 105, yd, cx + dx - 60, yB)
    add_label(s, cx - dx - 82, yd - 6, labA[0], color=labA[1])
    add_label(s, cx + dx + 14, yd - 6, labB[0], color=labB[1])
    return xA, xB


OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(OUT, "preview"), exist_ok=True)


def save(prs, name):
    prs.save(os.path.join(OUT, name))
    print("saved", name)


# ---------- 总体技术路线 ----------
def flow_route():
    W = 960
    prs = new_slide(W, 762)
    s = prs.slides[0]
    sz = 26
    _, h1 = add_box(s, 300, 30, 360, "读取题面、附件与接口约束", "io", sz)
    _, h2 = add_box(s, 280, 150, 400, "统一集合模型\n误差楔 · 距离域 · 清除域", "proc", sz)
    add_arrow(s, 480, 30 + h1, 480, 150)
    y3 = 150 + h2 + 54
    bw = 205
    t = ["问题一\n有界误差定位\n直径与覆盖判定", "问题二\n保证接收\n稳定交会选点",
         "问题三\n全向连续发现\n集合定位清除", "问题四\n定向连续发现\n正负观测调度"]
    xs = [22, 254, 486, 718]
    hs = []
    for x, txt in zip(xs, t):
        _, h = add_box(s, x, y3, bw, txt, "proc", sz)
        hs.append(h)
        add_arrow(s, 480, 150 + h2, x + bw / 2, y3)
    y4 = y3 + max(hs) + 50
    _, h5 = add_box(s, 180, y4, 600, "保守清除与滚动路线调度\n只在安全约束内压缩虚拟时间", "term", sz)
    for x in xs:
        tx = min(max(x + bw / 2, 250), 710)
        add_arrow(s, x + bw / 2, y3 + max(hs), tx, y4)
    y5 = y4 + h5 + 44
    add_box(s, 130, y5, 700, "分层验证：连续覆盖检验 · 离线合成回归\n官方演练锚定 · 正式测试定成绩", "term", sz)
    add_arrow(s, 480, y4 + h5, 480, y5)
    save(prs, "flow_route.pptx")


# ---------- 问题一 ----------
def flow_q1():
    W, bw = 860, 620
    items = [
        ("读取检测点与示向度（误差界 ±1°）", "io", None),
        ("每次观测转为闭角楔\n（两条线性半平面约束）", "proc", None),
        ("与目标圆、1500 m 距离圆盘求交\n（圆盘用内外正多边形夹逼）", "proc", None),
        ("可行域为空或无界？", "dec", None),
        ("占位A", "term", 28),
        ("枚举顶点对求直径\n并求最小覆盖圆（Welzl 类）", "proc", None),
        ("点积判据检验直径圆\n输出可复核的几何判定", "term", None),
    ]
    prs, ys, xc, _, hs = vertical_flow(W, items, bw=bw, gap=42, skip_auto=(3, 4))
    s = prs.slides[0]
    cx = xc + bw / 2
    delete_fullwidth_box(s, "占位A", bw)
    textA = "报告 EMPTY / UNBOUNDED\n不虚构定位结果"
    hA = box_h(textA, 28, "term")
    xA, xB = branch_pair(s, cx, ys[3][1], ys[4][0], hA, textA, ys[5][0] - 20, 0, "", dx=210, labA=("是", GREEN), labB=("否", RED))
    # A 为终态；B 侧占位不画框（branch_pair 画了空框，删除它）
    for sh in list(s.shapes):
        try:
            if sh.has_text_frame and sh.text_frame.text == "" and int(sh.width) == int(310 * PT):
                sh._element.getparent().remove(sh._element)
        except Exception:
            pass
    # 否分支直接进入框5（重新画箭头）
    add_arrow(s, cx + 105, ys[3][1], cx + 105, ys[5][0])
    save(prs, "flow_q1.pptx")


# ---------- 问题二 ----------
def flow_q2():
    W, bw = 860, 620
    items = [
        ("固定首测点与示向度\n建立一次观测可行域 F₁", "proc", None),
        ("坐标标准化\n构造 max‖p−z‖ ≤ 1000 保证接收域", "proc", None),
        ("网格枚举候选点\n计算连续最坏交会角", "proc", None),
        ("同时满足接收与 γ ≥ 30°？", "dec", None),
        ("占位A", "term", 28),
        ("占位B", "term", 28),
        ("两次观测交会定位\n保留全部误差相容位置", "io", None),
    ]
    prs, ys, xc, _, hs = vertical_flow(W, items, bw=bw, gap=40, skip_auto=(3, 4, 5))
    s = prs.slides[0]
    cx = xc + bw / 2
    delete_fullwidth_box(s, "占位A", bw)
    delete_fullwidth_box(s, "占位B", bw)
    tA, tB = "加密网格\n或报告无鲁棒候选", "按角度优先、路程次优\n选取第二检测点"
    hA, hB = box_h(tA, 28, "term"), box_h(tB, 28, "term")
    xA, xB = branch_pair(s, cx, ys[3][1], ys[4][0], hA, tA, ys[5][0], hB, tB, dx=205)
    ymA = max(ys[4][0] + hA, ys[5][0] + hB) + 18
    add_arrow(s, xA + 155, ys[4][0] + hA, xA + 155, ymA, head=False)
    add_arrow(s, xB + 155, ys[5][0] + hB, xB + 155, ymA, head=False)
    add_arrow(s, xA + 155, ymA, xB + 155, ymA, head=False)
    add_arrow(s, cx, ymA, cx, ys[6][0])
    save(prs, "flow_q2.pptx")


# ---------- 问题三 ----------
def flow_q3():
    W, bw = 880, 620
    items = [
        ("/enter：生成半径 998.925 m 的\n七点无圆心极小极大环", "io", None),
        ("按频道批处理访问扫描点\n完成未知频道发现", "proc", None),
        ("对正观测求角楔交\n与 1500 m 上界、目标圆求交", "proc", None),
        ("保守服务域半径 ≤ 19.5 m？", "dec", None),
        ("占位A", "term", 28),
        ("占位B", "term", 28),
        ("未收敛时：19.11 m 三角格覆盖兜底\n（必有一个格点距源小于 20 m）", "proc", None),
        ("滚动 2-opt 调度 → 全部清除 → /exit", "io", None),
    ]
    prs, ys, xc, _, hs = vertical_flow(W, items, bw=bw, gap=36, skip_auto=(3, 4, 5))
    s = prs.slides[0]
    cx = xc + bw / 2
    delete_fullwidth_box(s, "占位A", bw)
    delete_fullwidth_box(s, "占位B", bw)
    tA, tB = "选保证接收补测点\n更新可行域（回环）", "执行安全清除\n失败则保留可行域"
    hA, hB = box_h(tA, 28, "term"), box_h(tB, 28, "term")
    xA, xB = branch_pair(s, cx, ys[3][1], ys[4][0], hA, tA, ys[5][0], hB, tB, dx=205)
    ymA = max(ys[4][0] + hA, ys[5][0] + hB) + 16
    add_arrow(s, xA + 155, ys[4][0] + hA, xA + 155, ymA, head=False)
    add_arrow(s, xB + 155, ys[5][0] + hB, xB + 155, ymA, head=False)
    add_arrow(s, xA + 155, ymA, xB + 155, ymA, head=False)
    add_arrow(s, cx, ymA, cx, ys[6][0])
    save(prs, "flow_q3.pptx")


# ---------- 问题四 ----------
def flow_q4():
    W, bw = 880, 620
    items = [
        ("生成中心、八点内环（996 m）\n十二点外环（1863.756 m）共 21 点", "io", None),
        ("放大目标域（外切 720 边形）与 997 m\n保守接收圆作连续覆盖检验", "proc", None),
        ("存在未覆盖凸片或方向缺口？", "dec", None),
        ("占位A", "term", 28),
        ("扫描未知频道\n记录正观测与 no_signal 负观测", "proc", None),
        ("共同接收半径 ρ、方向弧 u 联合更新\n保守相容域（仅删严格不相容单元）", "proc", None),
        ("安全服务点 + 有限前瞻排序\n滚动 2-opt 与保证清除 → /exit", "io", None),
    ]
    prs, ys, xc, _, hs = vertical_flow(W, items, bw=bw, gap=38, skip_auto=(2, 3))
    s = prs.slides[0]
    cx = xc + bw / 2
    delete_fullwidth_box(s, "占位A", bw)
    tA = "调整半径或布局\n重新覆盖检验（回环）"
    hA = box_h(tA, 28, "term")
    xA, _ = branch_pair(s, cx, ys[2][1], ys[3][0], hA, tA, ys[4][0], 0, "", dx=205, labA=("是", RED), labB=("否", GREEN))
    for sh in list(s.shapes):
        try:
            if sh.has_text_frame and sh.text_frame.text == "" and int(sh.width) == int(310 * PT):
                sh._element.getparent().remove(sh._element)
        except Exception:
            pass
    add_arrow(s, cx + 105, ys[2][1], cx + 105, ys[4][0])
    # 回环：框3 左端 -> 框0 底部
    x_loop = xA + 60
    add_arrow(s, xA + 155, ys[3][0], x_loop, ys[3][0] - 2, head=False)
    add_arrow(s, x_loop, ys[3][0] - 2, x_loop, ys[0][1] + 12, head=False)
    add_arrow(s, x_loop, ys[0][1] + 12, cx, ys[0][1] + 12, head=False)
    add_arrow(s, cx, ys[0][1] + 12, cx, ys[0][1])
    save(prs, "flow_q4.pptx")


if __name__ == "__main__":
    flow_route(); flow_q1(); flow_q2(); flow_q3(); flow_q4()
    print("all done")
