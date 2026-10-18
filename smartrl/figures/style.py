"""Figure style. Sizes and IDs of RESULTS_SCHEMA; no caption inside the image.

`figures.language` in configs/frozen.yaml chooses the output:
- "en" (default, Roham 2026-10-18): English labels, Latin digits, Times New Roman.
- "fa": Persian labels with the Latin name in parentheses and Persian tick digits (PROJECT_RULES.md rule 9), B Nazanin.
  matplotlib >= 3.11 shapes Persian itself, so arabic_reshaper + python-bidi run only on older versions; the
  B Nazanin copy without brackets and hyphen avoids its legacy mirrored glyphs.
Plots call `label(en, fa)`, `market(m)`, `method(m)` and `ticks(ax, ...)` and never format text themselves.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ..config import FIGURES, ROOT, frozen, paths  # noqa: E402

FA_DIGITS = str.maketrans("0123456789.-", "۰۱۲۳۴۵۶۷۸۹٫−")
CM = 1 / 2.54
HEIGHT_CM = {"F02": 10, "F09": 10, "F10": 10, "F12": 12, "F14": 11}

MARKETS = {"SP500": ("S&P 500", "S&P 500"), "FTSE100": ("FTSE 100", "FTSE 100"), "NIFTY50": ("Nifty 50", "Nifty 50"),
           "SSE": ("SSE Composite", "شاخص کل بورس شانگهای (SSE Composite)"),
           "COMBINED": ("Combined", "سبد ترکیبی (Combined)")}
METHODS = {
    "SMARTRL": ("SMART-RL", "مدل پیشنهادی (SMART-RL)"), "B1_BH": ("Buy & Hold", "خرید و نگهداری (Buy & Hold)"),
    "B2_SMA": ("SMA 50/200", "میانگین متحرک ۵۰/۲۰۰ (SMA)"), "B3_TSMOM": ("TSMOM", "مومنتوم سری‌زمانی (TSMOM)"),
    "B4_LSTM": ("LSTM signal", "سیگنال LSTM (LSTM)"), "B5_FINBERT": ("FinBERT signal", "سیگنال FinBERT (FinBERT)"),
    "B6_PPOPRICE": ("PPO price-only", "PPO فقط قیمت (PPO price-only)"), "B7_SAC": ("SAC", "عامل SAC (SAC)"),
    "B8_STATIC": ("Static hybrid", "ترکیب ایستا (Static hybrid)"), "BASE": ("SenT-In (base paper)", "مدل مقاله‌ی مبنا (SenT-In)"),
    "CURR": ("CURR", "برنامه‌ی آموزشی (CURR)"), "PLAIN": ("PLAIN", "آموزش ساده (PLAIN)"),
    "BC": ("BC", "شروع گرم (BC)"), "IMIT": ("IMIT", "پاداش تقلیدی (IMIT)"),
}
BRACKETS = (0x28, 0x29, 0x5B, 0x5D, 0x7B, 0x7D, 0x2D)   # brackets and the hyphen
_ready: str | None = None


def language() -> str:
    return frozen()["figures"].get("language", "en")


def _native_shaping() -> bool:
    from packaging.version import Version
    return Version(matplotlib.__version__) >= Version("3.11")


def _nazanin_without_brackets(src: Path) -> Path:
    from fontTools.ttLib import TTFont
    from ..config import RAW
    dst = RAW / "fonts" / "BNazanin-figures.ttf"
    if not dst.exists():
        f = TTFont(str(src))
        for t in f["cmap"].tables:
            for cp in BRACKETS:
                t.cmap.pop(cp, None)
        dst.parent.mkdir(parents=True, exist_ok=True)
        f.save(str(dst))
    return dst


def setup() -> str:
    global _ready
    if _ready:
        return _ready
    cfg = frozen()["figures"]
    latin = [f for f in ("Times New Roman", "DejaVu Sans") if any(e.name == f for e in font_manager.fontManager.ttflist)]
    base = {"axes.unicode_minus": False, "pdf.fonttype": 42, "savefig.dpi": cfg["dpi"], "figure.dpi": 100,
            "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8}
    if language() == "fa":
        fonts = Path(paths()["fonts_dir"])
        cands = sorted(fonts.glob("*nazanin*.ttf")) + sorted(fonts.glob("*Nazanin*.ttf"))
        cands = [_nazanin_without_brackets(cands[0])] if cands else (sorted(fonts.glob("Vazirmatn*.ttf")) or sorted(
            (ROOT.parent / "thesis_writing" / "figures" / "diagrams").glob("Vazirmatn-Regular.ttf")))
        for f in cands:
            font_manager.fontManager.addfont(str(f))
        name = font_manager.FontProperties(fname=str(cands[0])).get_name()
        plt.rcParams.update({**base, **({"text.language": "fa"} if _native_shaping() else {}),
                             "font.family": [name] + latin})
        _ready = name
    else:
        plt.rcParams.update({**base, "font.family": latin or ["DejaVu Sans"]})
        _ready = (latin or ["DejaVu Sans"])[0]
    return _ready


def _shape(text: str) -> str:
    if _native_shaping() or not any("؀" <= ch <= "ۿ" for ch in text):
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    if text.endswith(")") and " (" in text:
        per, lat = text[:-1].rsplit(" (", 1)
        if not any("؀" <= ch <= "ۿ" for ch in lat):
            return f"({lat}) " + get_display(arabic_reshaper.reshape(per))
    return get_display(arabic_reshaper.reshape(text))


def label(en: str, fa: str) -> str:
    return _shape(fa) if language() == "fa" else en


def market(m: str) -> str:
    en, fa = MARKETS[m]
    return label(en, fa)


def method(m: str) -> str:
    en, fa = METHODS[m]
    return label(en, fa)


def ticks(ax, axis: str = "both", decimals: int | None = None, percent: bool = False) -> None:
    """Tick formatter: optional fixed decimals and percent; Persian digits only when language is 'fa'."""
    fa = language() == "fa"

    def f(x, _pos):
        v = x * 100 if percent else x
        s = f"{v:.{decimals}f}" if decimals is not None else f"{v:g}"
        if fa:
            s = s.translate(FA_DIGITS)
        return s + (("٪" if fa else "%") if percent else "")
    fmt = FuncFormatter(f)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def figure(fig_id: str, nrows: int = 1, ncols: int = 1, height_cm: float | None = None, **kw):
    setup()
    h = height_cm or HEIGHT_CM.get(fig_id, 10)
    return plt.subplots(nrows, ncols, figsize=(16 * CM, h * CM), **kw)


def save(fig, fig_id: str) -> list[Path]:
    """results/figures/<ID>.png (300 dpi) and .pdf; Persian versions go to results/figures/fa/."""
    d = FIGURES / "fa" if language() == "fa" else FIGURES
    d.mkdir(parents=True, exist_ok=True)
    out = [d / f"{fig_id}.png", d / f"{fig_id}.pdf"]
    fig.tight_layout()
    for p in out:
        fig.savefig(p, dpi=frozen()["figures"]["dpi"])
    plt.close(fig)
    return out
