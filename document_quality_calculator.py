"""
DOCUMENT QUALITY CALCULATOR
10 parameters for ALL document types + overall quality score.

Parameters:
  1.  Blur               — overall image sharpness (OpenCV)
  2.  Contrast           — light vs dark difference (OpenCV)
  3.  Brightness         — overall lightness/darkness (OpenCV)
  4.  Resolution         — pixel dimensions (OpenCV + PyMuPDF)
  5.  DPI                — dots per inch (PyMuPDF + Pillow)
  6.  Noise              — image graininess (Gaussian difference method)
  7.  Skew Angle         — how tilted the document is (OpenCV)
  8.  Shadow Coverage    — % covered by shadow (OpenCV)
  9.  Orientation        — portrait or landscape (OpenCV)
  10. Text Clarity       — sharpness of text/detail edges (OpenCV)

INSTALL:
  pip install opencv-python numpy pillow pymupdf
"""

import cv2
import numpy as np
from PIL import Image
import fitz
import os
from config import SUPPORTED_FORMATS


# ════════════════════════════════════════════════════════
# LOAD ANY DOCUMENT AS IMAGE
# ════════════════════════════════════════════════════════
def load_as_image(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        doc  = fitz.open(file_path)
        page = doc[0]
        mat  = fitz.Matrix(2.0, 2.0)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read: {file_path}")
        return img


# ════════════════════════════════════════════════════════
# PARAMETER 1 — BLUR
# Higher = sharper | Lower = blurrier
# ════════════════════════════════════════════════════════
def calculate_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 2 — CONTRAST
# Higher = better | Lower = washed out
# ════════════════════════════════════════════════════════
def calculate_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(float(gray.std()), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 3 — BRIGHTNESS
# Good range: 80–248
# White paper scans naturally score 230-248
# ════════════════════════════════════════════════════════
def calculate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(float(gray.mean()), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 4 — RESOLUTION
# PDFs → multiplied by 2 to convert points to pixels
# Images → actual pixel dimensions
# ════════════════════════════════════════════════════════
def calculate_resolution(file_path, img):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        doc  = fitz.open(file_path)
        page = doc[0]
        rect = page.rect
        zoom = 2
        return f"{int(rect.width * zoom)}x{int(rect.height * zoom)}"
    else:
        h, w = img.shape[:2]
        return f"{w}x{h}"


# ════════════════════════════════════════════════════════
# PARAMETER 5 — DPI
# PDFs → extract from embedded image metadata
# Images → EXIF first, then estimate from resolution
# ════════════════════════════════════════════════════════
def calculate_dpi(file_path, img):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        try:
            doc        = fitz.open(file_path)
            page       = doc[0]
            image_list = page.get_images(full=True)
            if image_list:
                for img_info in image_list:
                    xref       = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        img_w     = base_image.get("width", 0)
                        img_h     = base_image.get("height", 0)
                        page_w_in = page.rect.width / 72
                        page_h_in = page.rect.height / 72
                        if page_w_in > 0 and img_w > 0:
                            dpi_w = round(img_w / page_w_in)
                            dpi_h = round(img_h / page_h_in) if page_h_in > 0 else dpi_w
                            return round((dpi_w + dpi_h) / 2)
        except:
            pass
        return 144
    else:
        try:
            img_pil = Image.open(file_path)
            dpi     = img_pil.info.get('dpi', None)
            if dpi and dpi[0] > 72:
                return int(dpi[0])
        except:
            pass
        h, w   = img.shape[:2]
        aspect = max(w, h) / min(w, h)
        if 1.38 <= aspect <= 1.45:
            long_in, short_in = 11.69, 8.27
        elif 1.27 <= aspect <= 1.35:
            long_in, short_in = 11.0, 8.5
        else:
            long_in, short_in = 11.69, 8.27
        long_px  = max(w, h)
        short_px = min(w, h)
        return round((long_px / long_in + short_px / short_in) / 2)


# ════════════════════════════════════════════════════════
# PARAMETER 6 — NOISE  ← UPDATED: Gaussian Difference Method
#
# Method  : Subtract a Gaussian-blurred version from the
#           original to isolate high-frequency grain.
#           The std of that residual = noise level.
#
# Why better for documents:
#   estimate_sigma overestimates on high-contrast text edges.
#   Gaussian difference isolates TRUE grain from sharp strokes,
#   making it far more accurate for scanned/photographed docs.
#
# Kernel  : (5, 5) with sigmaX=1.0 — removes grain but
#           preserves text/line structure
#
# Lower   = cleaner | Higher = more grainy
# ════════════════════════════════════════════════════════
def calculate_noise(img):
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred  = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)
    residual = gray - blurred                       # pure grain / noise signal
    return round(float(residual.std()), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 7 — SKEW ANGLE
# Lower = straighter | Higher = more tilted
# ════════════════════════════════════════════════════════
def calculate_skew_angle(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_not(gray)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return round(abs(angle), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 8 — SHADOW COVERAGE
# Returns % of document covered by shadow
# ════════════════════════════════════════════════════════
def calculate_shadow_coverage(img):
    gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    background = cv2.GaussianBlur(gray, (51, 51), 0)
    bg_max     = background.max()
    if bg_max == 0:
        return 0.0
    shadow_mask = background < (bg_max * 0.7)
    return round(float(shadow_mask.sum() / gray.size * 100), 2)


# ════════════════════════════════════════════════════════
# PARAMETER 9 — ORIENTATION
# Returns portrait or landscape
# ════════════════════════════════════════════════════════
def calculate_orientation(img):
    h, w = img.shape[:2]
    return "landscape" if w > h else "portrait"



# ════════════════════════════════════════════════════════
# PARAMETER 11 — TEXT CLARITY
# 95th percentile of Sobel edge magnitude
# Higher = sharper text/details
# ════════════════════════════════════════════════════════
def calculate_text_clarity(img):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag    = np.sqrt(edge_x**2 + edge_y**2)
    return round(float(np.percentile(mag, 95)), 2)


# ════════════════════════════════════════════════════════
# NORMALIZE HELPERS
# Each parameter converted to a 0–100 component score
# before weighting. This ensures no single raw scale
# (e.g. blur=2000 vs noise=5) dominates unfairly.
# ════════════════════════════════════════════════════════
def _clamp(val, lo, hi):
    """Clamp val between lo and hi, then scale to 0–100."""
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))


def _normalize_blur(blur):
    # <200 = very blurry → 0 | 1500+ = sharp → 100
    return _clamp(blur, 200, 1500)


def _normalize_contrast(contrast):
    # <20 = washed out → 0 | 80+ = high contrast → 100
    return _clamp(contrast, 20, 80)


def _normalize_brightness(brightness):
    # Ideal band: 80–248. Penalise both extremes with a tent function.
    if brightness < 80:
        return _clamp(brightness, 40, 80)
    elif brightness <= 248:
        return 100.0
    else:
        return _clamp(255 - brightness, 0, 7)   # >248 quickly drops


def _normalize_noise(noise):
    # Gaussian-difference noise: <2 = very clean | >20 = very noisy
    # Invert so low noise → high score
    return _clamp(20 - noise, 0, 18)            # score = 100 at noise≤2


def _normalize_skew(skew):
    # 0° = perfect | >5° = bad
    return _clamp(5 - skew, 0, 5)


def _normalize_shadow(shadow):
    # 0% = no shadow (perfect) | >30% = heavy shadow
    return _clamp(30 - shadow, 0, 30)


def _normalize_dpi(dpi):
    # <72 = very low | 300+ = excellent
    return _clamp(dpi, 72, 300)


def _normalize_clarity(clarity):
    # <5 = unreadable | 200+ = extremely sharp
    return _clamp(clarity, 5, 200)


# ════════════════════════════════════════════════════════
# OVERALL QUALITY SCORE  ← UPDATED: Weighted Average
#
# Why weights instead of penalties:
#   • Penalties are order-dependent and can stack unfairly.
#   • A weighted average gives a fair, normalized 0–100 score
#     where each parameter contributes proportionally.
#   • Weights reflect what matters most for OCR / extraction:
#     blur and text clarity together account for 40% because
#     a blurry or low-clarity document fails OCR regardless
#     of how good everything else is.
#
# WEIGHTS (must sum to 1.0):
#   Blur             0.20  — primary sharpness driver
#   Text Clarity     0.20  — OCR / extraction quality
#   Contrast         0.15  — legibility of strokes
#   Noise            0.15  — grain hides fine details
#   Brightness       0.10  — under/over-exposed kills reads
#   Skew             0.10  — tilted docs confuse layout parsers
#   Shadow           0.05  — localised darkness
#   DPI              0.05  — sensor/scan resolution
#
# Parameters without a numeric score (resolution, orientation)
# are categorical — they are reported but not included in the
# weighted calculation.
# ════════════════════════════════════════════════════════

WEIGHTS = {
    "blur":      0.20,
    "clarity":   0.20,
    "contrast":  0.15,
    "noise":     0.15,
    "brightness":0.10,
    "skew":      0.10,
    "shadow":    0.05,
    "dpi":       0.05,
}
# Sanity check — weights must sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


def calculate_overall_quality(blur, contrast, brightness,
                               resolution, dpi, noise,
                               skew, shadow, clarity):

    components = {
        "blur":       _normalize_blur(blur),
        "clarity":    _normalize_clarity(clarity),
        "contrast":   _normalize_contrast(contrast),
        "noise":      _normalize_noise(noise),
        "brightness": _normalize_brightness(brightness),
        "skew":       _normalize_skew(skew),
        "shadow":     _normalize_shadow(shadow),
        "dpi":        _normalize_dpi(dpi),
    }

    score = sum(WEIGHTS[k] * v for k, v in components.items())
    score = round(score, 1)

    if score >= 80:
        label = "GOOD"
    elif score >= 60:
        label = "MODERATE"
    elif score >= 40:
        label = "POOR"
    else:
        label = "VERY POOR"

    return score, label, components   # components returned for debugging


# ════════════════════════════════════════════════════════
# SCAN ALL FOLDERS
# ════════════════════════════════════════════════════════
def scan_folders(folder_paths):
    all_files = []
    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"  ⚠️  Folder not found: {folder}")
            continue
        for root, dirs, files in os.walk(folder):
            for file in sorted(files):
                if file.endswith(SUPPORTED_FORMATS):
                    all_files.append({
                        "folder":      folder,
                        "folder_name": os.path.basename(folder),
                        "file_path":   os.path.join(root, file),
                        "file_name":   file,
                    })
    return all_files


# ════════════════════════════════════════════════════════
# ANALYZE ONE DOCUMENT
# ════════════════════════════════════════════════════════
def analyze_document(file_info):
    file_path   = file_info["file_path"]
    img         = load_as_image(file_path)
    resolution  = calculate_resolution(file_path, img)
    orientation = calculate_orientation(img)
    blur        = calculate_blur(img)
    contrast    = calculate_contrast(img)
    brightness  = calculate_brightness(img)
    dpi         = calculate_dpi(file_path, img)
    noise       = calculate_noise(img)
    skew        = calculate_skew_angle(img)
    shadow      = calculate_shadow_coverage(img)
    clarity     = calculate_text_clarity(img)


    overall, label, components = calculate_overall_quality(
        blur, contrast, brightness,
        resolution, dpi, noise,
        skew, shadow, clarity
    )

    return {
        "folder_name":           file_info["folder_name"],
        "file_name":             file_info["file_name"],
        "blur":                  blur,
        "contrast":              contrast,
        "brightness":            brightness,
        "resolution":            resolution,
        "dpi":                   dpi,
        "noise":                 noise,
        "skew_angle_deg":        skew,
        "shadow_coverage_pct":   shadow,
        "orientation":           orientation,
        "text_clarity":          clarity,
        "overall_quality_score": overall,
        "overall_quality":       label,
        "quality_components":    components,
    }


# ════════════════════════════════════════════════════════
# PRINT SCORES
# ════════════════════════════════════════════════════════
def print_scores(r):
    c = r.get("quality_components", {})
    print(f"\n  {'='*60}")
    print(f"  File     : {r['file_name']}")
    print(f"  Folder   : {r['folder_name']}")
    print(f"  {'─'*60}")
    print(f"  {'Blur':<28} {r['blur']:<12}  component: {c.get('blur', '—'):.1f}/100")
    print(f"  {'Contrast':<28} {r['contrast']:<12}  component: {c.get('contrast', '—'):.1f}/100")
    print(f"  {'Brightness':<28} {r['brightness']:<12}  component: {c.get('brightness', '—'):.1f}/100")
    print(f"  {'Resolution':<28} {r['resolution']}")
    print(f"  {'DPI':<28} {r['dpi']:<12}  component: {c.get('dpi', '—'):.1f}/100")
    print(f"  {'Noise':<28} {r['noise']:<12}  component: {c.get('noise', '—'):.1f}/100")
    print(f"  {'Skew Angle':<28} {r['skew_angle_deg']}°{'':<9}  component: {c.get('skew', '—'):.1f}/100")
    print(f"  {'Shadow Coverage':<28} {r['shadow_coverage_pct']}%{'':<9}  component: {c.get('shadow', '—'):.1f}/100")
    print(f"  {'Orientation':<28} {r['orientation']}")
    print(f"  {'Text Clarity':<28} {r['text_clarity']:<12}  component: {c.get('clarity', '—'):.1f}/100")

    print(f"  {'─'*60}")
    print(f"  {'Overall Quality':<28} {r['overall_quality_score']}/100"
          f"  → {r['overall_quality']}")
    print(f"  {'='*60}")