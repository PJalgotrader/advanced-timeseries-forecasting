"""Puts the standard header cell (banner, Colab button, video and links) at the top of a notebook and of its source.

  python tools/notebook_header.py 07-deep-sequence-models/01_rnn_lstm_one_series.ipynb https://youtu.be/<video id>

Running it again replaces the header instead of adding a second one. The notebook's saved outputs are kept, so it
does not have to be run again. Change the links here, then run it for every notebook. Needs no packages.
"""
import json, os, sys
from urllib.parse import quote

REPO = "PJalgotrader/advanced-timeseries-forecasting"
MARK = "<!-- notebook-header -->"
BANNER = f"https://raw.githubusercontent.com/{REPO}/main/assets/header/banner.png"
PLAYLIST = "https://www.youtube.com/playlist?list=PLG1FN7Dkkeu8"   # Deep Forecasting 2027


def badge(label, logo, url, gold=False):
    color, logo_color = ("D0A63C", "0A1020") if gold else ("0A1020", "E8C468")
    img = f"https://img.shields.io/badge/{quote(label.replace('-', '--').replace(' ', '_'))}-{color}?style=for-the-badge&logo={logo}&logoColor={logo_color}"
    return f'<a href="{url}"><img src="{img}" alt="{label}"></a>'


def header(nb_path, video):
    colab = f"https://colab.research.google.com/github/{REPO}/blob/main/{quote(nb_path)}"
    badges = " ".join([
        badge("Watch the video", "youtube", video, gold=True),
        badge("Deep Forecasting 2027", "youtube", PLAYLIST),
        badge("pedramjahangiry.com", "googlechrome", "https://pedramjahangiry.com"),
        badge("Newsletter", "substack", "https://pedramjahangiry.com/newsletter.html"),
        badge("YouTube", "youtube", "https://www.youtube.com/@pedramjahangiry"),
        badge("LinkedIn", "linkedin", "https://www.linkedin.com/in/pedram-jahangiry-cfa-5778015a"),
        badge("@pjalgo", "x", "https://x.com/pjalgo"),
        badge("GitHub", "github", "https://github.com/PJalgotrader"),
    ])
    return [
        MARK,
        f'<a href="https://pedramjahangiry.com"><img src="{BANNER}" alt="Dr. Pedram Jahangiry: ML, deep learning and time series, explained visually" width="100%"></a>',
        "",
        f'<a href="{colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
        " &nbsp; Run it in Colab. To keep your changes, choose **File → Save a copy in Drive** first.",
        "",
        badges,
        "",
        "---",
    ]


def main():
    nb_file, video = sys.argv[1], sys.argv[2]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel = os.path.relpath(os.path.abspath(nb_file), root).replace(os.sep, "/")
    lines = header(rel, video)

    nb = json.load(open(nb_file, encoding="utf-8"))
    cell = {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if nb["cells"] and "".join(nb["cells"][0]["source"]).startswith(MARK): nb["cells"][0] = cell
    else: nb["cells"].insert(0, cell)
    json.dump(nb, open(nb_file, "w", encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)

    src = os.path.join(os.path.dirname(nb_file), "src", os.path.basename(nb_file)[:-6] + ".py")
    text = open(src, encoding="utf-8").read()
    block = "# %% [markdown]\n" + "\n".join(("# " + l) if l else "#" for l in lines) + "\n\n"
    if text.startswith("# %% [markdown]\n# " + MARK):
        text = text[text.index("# %%", 1):]
    open(src, "w", encoding="utf-8", newline="\n").write(block + text)
    print("header ->", nb_file, "and", src)


if __name__ == "__main__":
    main()
