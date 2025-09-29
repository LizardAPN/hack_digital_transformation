import sys
from pathlib import Path

# Добавляем путь к утилитам для корректной работы импортов
utils_path = Path(__file__).resolve().parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Настраиваем пути проекта
try:
    from path_resolver import setup_project_paths
    setup_project_paths()
except ImportError:
    # Если path_resolver недоступен, добавляем необходимые пути вручную
    src_path = Path(__file__).resolve().parent.parent
    paths_to_add = [src_path, src_path / "utils", src_path / "geo"]
    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
import re
from typing import List, Optional, Tuple

import cv2
import easyocr
import numpy as np


class OverlayOCR:
    WHITELIST_RE = re.compile(r"[A-Za-z0-9_]+")

    def __init__(
        self,
        langs: Optional[List[str]] = None,
        gpu: bool = False,
        verbose: bool = False,
        gap_mult: float = 1.6,
        canvas_size: int = 3600,
        mag_ratio: float = 3.0,
        add_margin: float = 0.10,
        text_threshold: float = 0.55,
        low_text: float = 0.30,
        link_threshold: float = 0.30,
    ):
        """
        langs: языки easyocr, напр. ['en'] или ['en','ru']
        gap_mult: чувствительность к горизонтальным разрывам (меньше -> больше '_')
        canvas_size/mag_ratio: масштабирование внутри easyocr
        """
        self.langs = langs or ["en"]
        self.reader = easyocr.Reader(self.langs, gpu=gpu, verbose=verbose)
        self.gap_mult = gap_mult
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.add_margin = add_margin
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold

    # ---------- утилиты ----------
    @staticmethod
    def _clean_token(t: str) -> str:
        return "".join(OverlayOCR.WHITELIST_RE.findall(t))

    @staticmethod
    def _alnum_class(ch: str) -> str:
        return "D" if ch.isdigit() else ("A" if ch.isalpha() else "_")

    def _join_with_gaps(self, results, sep="_") -> Tuple[str, float, list]:
        """
        Склейка токенов слева направо:
        - '_' если горизонтальный зазор >> медианного,
        - '_' на границах A<->D.
        """
        items = []
        for bbox, text, conf in results:
            t = self._clean_token(text)
            if not t:
                continue
            x0 = min(p[0] for p in bbox)
            x1 = max(p[0] for p in bbox)
            items.append((x0, x1, t, float(conf)))
        if not items:
            return "", 0.0, []

        items.sort(key=lambda z: z[0])
        gaps = []
        for i in range(1, len(items)):
            gaps.append(items[i][0] - items[i - 1][1])
        med_gap = np.median(gaps) if gaps else 0

        out = []
        confs = []
        prev = None
        for i, (x0, x1, t, c) in enumerate(items):
            if prev is not None:
                gap = x0 - prev[1]
                need_sep = med_gap > 0 and gap > self.gap_mult * med_gap
                # буква↔️цифра – полезно отделить
                if not need_sep:
                    prev_last = out[-1][-1] if out else ""
                    if prev_last and t:
                        need_sep = self._alnum_class(prev_last) != self._alnum_class(t[0])
                if need_sep and (not out or out[-1] != sep):
                    out.append(sep)
            out.append(t)
            confs.append(c)
            prev = (x0, x1)

        text = "".join(out)
        text = re.sub(r"_+", "_", text).strip("_")
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return text, avg_conf, items

    @staticmethod
    def _normalize_overlays(s: str) -> str:
        """Правки под формат MMC_hd_... и расстановка подчёркиваний."""
        s = re.sub(r"^MMC(?:_)?h(?:d)?", "MMC_hd", s, flags=re.IGNORECASE)
        s = re.sub(r"^MMC_?hd_?", "MMC_hd_", s, flags=re.IGNORECASE)
        s = re.sub(r"([A-Za-z])([0-9])", r"\1_\2", s)
        s = re.sub(r"([0-9])([A-Za-z])", r"\1_\2", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    @staticmethod
    def _snap_digits_tail(s: str) -> str:
        """
        Если хвост цифр склеен, режем на 4-1-1 (типичный случай).
        Пример: ...229221 -> ...2292_2_1
        """
        m = re.search(r"^(.*?)(\d{6,})$", s)
        if not m:
            return s
        head, digits = m.group(1), m.group(2)
        if len(digits) >= 6:
            s = f"{head}{digits[:4]}_{digits[4:5]}_{digits[5:]}"
        return re.sub(r"_+", "_", s).strip("_")

    # ---------- EasyOCR запуск на ROI ----------
    def run_on_roi(self, roi_bgr) -> Tuple[str, str, str, float]:
        params = dict(
            decoder="greedy",
            detail=1,
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=self.text_threshold,
            low_text=self.low_text,
            link_threshold=self.link_threshold,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
            add_margin=self.add_margin,
        )
        results = self.reader.readtext(roi_bgr, **params)
        joined, conf, _ = self._join_with_gaps(results, sep="_")
        norm = self._normalize_overlays(joined)
        final = self._snap_digits_tail(norm)
        final = re.sub(r"^MMC_?hd_?", "MMC_hd_", final, flags=re.IGNORECASE)
        final = re.sub(r"_+", "_", final).strip("_")
        return joined, norm, final, conf

    # ---------- ROI генераторы ----------
    @staticmethod
    def roi_left_bottom(img, w_frac=1 / 3, h_frac=1 / 4):
        H, W = img.shape[:2]
        return img[H - int(H * h_frac) : H, 0 : int(W * w_frac)]

    @staticmethod
    def roi_bottom_band(img, h_frac=1 / 3):
        H, _ = img.shape[:2]
        y0 = H - int(H * h_frac)
        return img[y0:H, :]

    @staticmethod
    def roi_auto_band(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        row = (b > 0).sum(axis=1).astype(np.float32)
        k = max(3, (img.shape[0] // 100) * 2 + 1)
        row = cv2.GaussianBlur(row.reshape(-1, 1), (1, k), 0).ravel()
        start = img.shape[0] // 2
        idx = start + int(np.argmax(row[start:]))
        band_half = max(img.shape[0] // 12, 20)
        y0, y1 = max(0, idx - band_half), min(img.shape[0], idx + band_half)
        return img[y0:y1, :]

    # ---------- главный метод ----------
    def run_on_image(self, image_path: str) -> Tuple[str, str, str, float, str]:
        """
        Возвращает:
          final, norm, joined, conf, best_roi_name
        """
        img = cv2.imread(image_path)
        assert img is not None, f"Не удалось загрузить изображение: {image_path}"

        rois = [
            ("left_bottom", self.roi_left_bottom(img, 1 / 3, 1 / 4)),
            ("bottom_band", self.roi_bottom_band(img, 1 / 3)),
            ("auto_band", self.roi_auto_band(img)),
        ]

        best = None
        best_name = ""
        best_pack = ("", "", "", 0.0)

        for name, roi in rois:
            joined, norm, final, conf = self.run_on_roi(roi)
            cand = (conf, len(final), (final, norm, joined, conf), name)
            if (best is None) or (cand > best):
                best = cand
                best_pack = (final, norm, joined, conf)
                best_name = name

        final, norm, joined, conf = best_pack
        return final, norm, joined, conf, best_name
