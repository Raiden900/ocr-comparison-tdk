import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract, easyocr
import difflib
import os, json, tempfile
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# --------------------------------------------------
# MODELLEK BETÖLTÉSE
# --------------------------------------------------

reader = easyocr.Reader(['en'], gpu=False)
doctr_model = ocr_predictor(pretrained=True)

# Ha kell Windows alatt:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------------------------------
# SEGÉDFÜGGVÉNYEK
# --------------------------------------------------

def show(img, title=None, size=(6,6)):
    plt.figure(figsize=size)
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def normalize(text):
    text = text.replace("\x0c", "")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.upper().strip()

def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# --------------------------------------------------
# KÉP BETÖLTÉS
# --------------------------------------------------

Tk().withdraw()

print("Válaszd ki a képet:")
img_path = askopenfilename(
    title="Kép kiválasztása",
    filetypes=[("Képfájlok", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not img_path:
    raise ValueError("Nem választottál ki képet!")

data = np.fromfile(img_path, dtype=np.uint8)
img = cv.imdecode(data, cv.IMREAD_COLOR)

show(img, "Eredeti fotó", (8,8))

# --------------------------------------------------
# PERSPEKTÍVA JAVÍTÁS
# --------------------------------------------------

def largest_quad_contour(binary):
    cnts,_ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best=None; area=0
    for c in cnts:
        peri=cv.arcLength(c,True)
        approx=cv.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            a=cv.contourArea(approx)
            if a>area: best=approx; area=a
    return best

def order_pts(pts):
    pts=pts.reshape(4,2).astype(np.float32)
    s=pts.sum(1); d=np.diff(pts, axis=1).ravel()
    tl=pts[np.argmin(s)]; br=pts[np.argmax(s)]
    tr=pts[np.argmin(d)]; bl=pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], np.float32)

def warp_document(image_bgr, target_width=1200):
    ratio = target_width / image_bgr.shape[1]
    img2 = cv.resize(image_bgr, (target_width, int(image_bgr.shape[0]*ratio)))
    gray=cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray=cv.GaussianBlur(gray,(5,5),0)
    edges=cv.Canny(gray,60,180)
    edges=cv.dilate(edges,np.ones((3,3),np.uint8),1)
    quad=largest_quad_contour(edges)
    if quad is None:
        return img2, False
    pts=order_pts(quad)
    w=int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    h=int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    M=cv.getPerspectiveTransform(pts, np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32))
    warped=cv.warpPerspective(img2, M, (w,h))
    return warped, True

raw, ok = warp_document(img)
show(raw, "RAW")

# --------------------------------------------------
# ELŐFELDOLGOZÁS
# --------------------------------------------------

def enhance_for_ocr(bgr):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)
    gray = cv.convertScaleAbs(gray, alpha=1.15, beta=0)
    _, binimg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return gray, binimg



gray, binimg = enhance_for_ocr(raw)
show(gray, "PROC – szürke")
show(binimg, "PROC")

# --------------------------------------------------
# OCR FÜGGVÉNYEK
# --------------------------------------------------

def ocr_easy(img_gray):
    res = reader.readtext(
        img_gray,
        detail=0,
        paragraph=True,
        decoder='beamsearch',
        text_threshold=0.4,
        low_text=0.3
    )
    return "\n".join(res)

def ocr_tess(img_gray):
    cfg = "--oem 1 --psm 4 -l eng"
    return pytesseract.image_to_string(img_gray, config=cfg)

def ocr_doctr(image_bgr):
    tmp_path = os.path.join(tempfile.gettempdir(), "doctr_tmp.jpg")
    cv.imwrite(tmp_path, image_bgr)
    doc = DocumentFile.from_images(tmp_path)
    result = doctr_model(doc)

    lines = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words]
                lines.append(" ".join(words))
    return "\n".join(lines)

# --------------------------------------------------
# OCR FUTTATÁS
# --------------------------------------------------

txt_easy_raw  = ocr_easy(cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
txt_easy_proc = ocr_easy(binimg)

txt_tess_raw  = ocr_tess(cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
txt_tess_proc = ocr_tess(binimg)

txt_doctr_raw  = ocr_doctr(raw)
txt_doctr_proc = ocr_doctr(cv.cvtColor(binimg, cv.COLOR_GRAY2BGR))

# --------------------------------------------------
# KIÉRTÉKELÉS
# --------------------------------------------------

expected = input("\nÍrd be az elvárt szöveget:\n")

norm_expected = normalize(expected)

norm_easy_raw  = normalize(txt_easy_raw)
norm_easy_proc = normalize(txt_easy_proc)
norm_tess_raw  = normalize(txt_tess_raw)
norm_tess_proc = normalize(txt_tess_proc)
norm_doctr_raw  = normalize(txt_doctr_raw)
norm_doctr_proc = normalize(txt_doctr_proc)

results = {
    "EasyOCR RAW": similarity(norm_expected, norm_easy_raw),
    "EasyOCR PROC": similarity(norm_expected, norm_easy_proc),
    "Tesseract RAW": similarity(norm_expected, norm_tess_raw),
    "Tesseract PROC": similarity(norm_expected, norm_tess_proc),
    "docTR RAW": similarity(norm_expected, norm_doctr_raw),
    "docTR PROC": similarity(norm_expected, norm_doctr_proc),
}

print("\n" + "="*70)
print("PONTOSSÁGI EREDMÉNYEK")
print("="*70)

for name, val in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:20s} -> {val*100:6.2f}%")

print("="*70)

# --------------------------------------------------
# MENTÉS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

with open("outputs/all_ocr_results.json","w",encoding="utf-8") as f:
    json.dump({
        "expected": expected,
        "results": results
    }, f, indent=2)

print("\nMentve: outputs/all_ocr_results.json")
