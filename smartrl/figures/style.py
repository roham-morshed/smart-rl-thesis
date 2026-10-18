"""Figure style of PROJECT_RULES.md rule 9: Persian labels (Latin name in parentheses), Persian tick digits, B Nazanin
(fallback Vazirmatn), shaped with arabic_reshaper + python-bidi, no caption in the image, sizes of RESULTS_SCHEMA."""
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

MARKET_FA = {"SP500": "S&P 500", "FTSE100": "FTSE 100", "NIFTY50": "Nifty 50",
             "SSE": "شاخص کل بورس شانگهای (SSE Composite)", "COMBINED": "سبد ترکیبی (Combined)"}
METHOD_FA = {
    "SMARTRL": "مدل پیشنهادی (SMART-RL)", "B1_BH": "خرید و نگهداری (Buy & Hold)", "B2_SMA": "میانگین متحرک ۵۰/۲۰۰ (SMA)",
    "B3_TSMOM": "مومنتوم سری‌زمانی (TSMOM)", "B4_LSTM": "سیگنال LSTM (LSTM)", "B5_FINBERT": "سیگنال FinBERT (FinBERT)",
    "B6_PPOPRICE": "PPO فقط قیمت (PPO price-only)", "B7_SAC": "عامل SAC (SAC)", "B8_STATIC": "ترکیب ایستا (Static hybrid)",
    "BASE": "مدل مقاله‌ی مبنا (SenT-In)", "CURR": "برنامه‌ی آموزشی (CURR)", "PLAIN": "آموزش ساده (PLAIN)",
    "BC": "شروع گرم (BC)", "IMIT": "پاداش تقلیدی (IMIT)",
}

_font_name: str | None = None
BRACKETS = (0x28, 0x29, 0x5B, 0x5D, 0x7B, 0x7D, 0x2D)   # brackets and the hyphen


def _nazanin_without_brackets(src: Path) -> Path:
    """Local copy of B Nazanin without bracket characters in its cmap.

    The legacy B Nazanin draws '(' and ')' for the old right-to-left behaviour, so with matplotlib's native
    bidi layout a closing parenthesis comes out mirrored. Without them in the cmap, matplotlib takes brackets
    from the Latin fallback font; every Persian glyph still comes from B Nazanin."""
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
    """Register B Nazanin (or Vazirmatn) and return the family name in use."""
    global _font_name
    if _font_name:
        return _font_name
    cfg = frozen()["figures"]
    fonts = Path(paths()["fonts_dir"])
    cands = sorted(fonts.glob("*nazanin*.ttf")) + sorted(fonts.glob("*Nazanin*.ttf"))
    if cands:
        cands = [_nazanin_without_brackets(cands[0])]
    else:
        cands = sorted(fonts.glob("Vazirmatn*.ttf")) or sorted(
            (ROOT.parent / "thesis_writing" / "figures" / "diagrams").glob("Vazirmatn-Regular.ttf"))
    for f in cands:
        font_manager.fontManager.addfont(str(f))
    _font_name = font_manager.FontProperties(fname=str(cands[0])).get_name()
    # B Nazanin has no Latin glyphs: matplotlib takes each missing glyph from the next family in the list
    latin = [f for f in ("Times New Roman", "DejaVu Sans") if any(e.name == f for e in font_manager.fontManager.ttflist)]
    # text.language "fa" makes the native layout run the bidi algorithm with a right-to-left base and mirror
    # neutral characters, so "persian (Latin)" keeps its parentheses the right way round
    lang = {"text.language": "fa"} if _native_shaping() else {}
    plt.rcParams.update({**lang, "font.family": [_font_name] + latin, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "savefig.dpi": cfg["dpi"], "figure.dpi": 100, "axes.titlesize": 11, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9})
    return _font_name


def _native_shaping() -> bool:
    """matplotlib >= 3.11 lays out complex scripts itself (letter joining and right-to-left order); shaping the
    string beforehand with arabic_reshaper + python-bidi would then reverse it a second time."""
    from packaging.version import Version
    return Version(matplotlib.__version__) >= Version("3.11")


def _shape(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def fa(text: str) -> str:
    """Persian label for matplotlib. With native shaping the string is passed unchanged; otherwise it is shaped with
    arabic_reshaper + python-bidi, and 'persian (Latin)' is composed in visual order as '(Latin) <persian>' so the
    bidi algorithm cannot move or mirror the parentheses. Text without Persian letters is left as it is."""
    if _native_shaping() or not any("؀" <= ch <= "ۿ" for ch in text):
        return text
    if text.endswith(")") and " (" in text:
        per, lat = text[:-1].rsplit(" (", 1)
        if not any("؀" <= ch <= "ۿ" for ch in lat):
            return f"({lat}) " + _shape(per)
    return _shape(text)


def fa_num(x, decimals: int | None = None) -> str:
    s = f"{x:.{decimals}f}" if decimals is not None else (f"{x:g}")
    return s.translate(FA_DIGITS)


def persian_ticks(ax, axis: str = "both", decimals: int | None = None, percent: bool = False) -> None:
    def f(x, _pos):
        v = x * 100 if percent else x
        s = fa_num(v, decimals)
        return s + ("٪" if percent else "")
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
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = [FIGURES / f"{fig_id}.png", FIGURES / f"{fig_id}.pdf"]
    fig.tight_layout()
    for p in out:
        fig.savefig(p, dpi=frozen()["figures"]["dpi"])
    plt.close(fig)
    return out
