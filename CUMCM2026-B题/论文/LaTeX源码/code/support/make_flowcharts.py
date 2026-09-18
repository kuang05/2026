from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "figures"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def make_flow(filename, stages):
    OUT.mkdir(parents=True, exist_ok=True)
    count = len(stages)
    fig, ax = plt.subplots(figsize=(14, 3.3))
    ax.set_xlim(0, count)
    ax.set_ylim(0, 1)
    ax.axis("off")
    colors = ["#dceaf7", "#e5f2e6", "#f7ead7", "#e8e2f4", "#f6e0e0"]

    for index, (title, detail) in enumerate(stages):
        x = index + 0.5
        ax.text(
            x,
            0.89,
            title,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        box = FancyBboxPatch(
            (x - 0.39, 0.23),
            0.78,
            0.42,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.6,
            edgecolor="#404040",
            facecolor=colors[index % len(colors)],
        )
        ax.add_patch(box)
        ax.text(
            x,
            0.44,
            detail,
            ha="center",
            va="center",
            fontsize=16,
            linespacing=1.25,
        )
        if index < count - 1:
            arrow = FancyArrowPatch(
                (x + 0.40, 0.44),
                (x + 0.60, 0.44),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color="#606060",
            )
            ax.add_patch(arrow)

    fig.savefig(OUT / filename, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    make_flow(
        "flow-overall.pdf",
        [
            ("题面输入", "误差、半径\n计时与边界"),
            ("保守定位", "闭角楔\n相容位置域"),
            ("连续发现", "圆覆盖\n正向包围"),
            ("安全清除", "覆盖圆\n三角格兜底"),
            ("时间调度", "滚动路线\n有限前瞻"),
        ],
    )
    make_flow(
        "flow-q1.pdf",
        [
            ("读取观测", "检测点与\n示向度"),
            ("构造角楔", "有界误差\n半平面"),
            ("求相容域", "交会定位\n距离收紧"),
            ("几何计算", "直径与\n最小覆盖圆"),
            ("输出核验", "点积判据\n反例结论"),
        ],
    )
    make_flow(
        "flow-q2.pdf",
        [
            ("首测归一化", "平移与\n旋转坐标"),
            ("保证接收", "所有可能源\n均距不超1000"),
            ("角度筛选", "连续最坏\n交会角"),
            ("候选评估", "网格计算\n距离与角度"),
            ("确定第二点", "排序后\n反变换"),
        ],
    )
    make_flow(
        "flow-q3.pdf",
        [
            ("七点环覆盖", "解析极小极大\n连续发现"),
            ("轮询频道", "未发现频道\n完整扫描"),
            ("更新域", "角楔交\n安全补测"),
            ("保证清除", "服务区域\n三角格兜底"),
            ("滚动路线", "批量访问\n2-opt调度"),
        ],
    )
    make_flow(
        "flow-q4.pdf",
        [
            ("正向包围", "21点扫描集\n连续核验"),
            ("联合观测", "正负反馈\n相容单元"),
            ("有限前瞻", "候选假设\n时间排序"),
            ("安全服务", "保证域内\n短路选点"),
            ("失败保护", "测向恢复\n三角格兜底"),
        ],
    )


if __name__ == "__main__":
    main()
