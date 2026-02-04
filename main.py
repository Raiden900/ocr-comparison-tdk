# Szükséges csomagok egyszer kell telepíteni:
# pip install opencv-python-headless easyocr pytesseract matplotlib numpy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract, easyocr
import difflib
import os, json
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def show(img, title=None, size=(6,6)):
    plt.figure(figsize=size)
    if img.ndim==2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# --- KÉP BETÖLTÉSE PC-N ---

Tk().withdraw()

print("Válaszd ki a képet:")
img_path = askopenfilename(
    title="Kép kiválasztása",
    filetypes=[("Képfájlok", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

if not img_path:
    raise ValueError("Nem választottál ki képet!")

# Unicode-barát betöltés, ékezetes útvonalakhoz is jó
data = np.fromfile(img_path, dtype=np.uint8)
img = cv.imdecode(data, cv.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"Nem sikerült betölteni a képet: {img_path}")

show(img, "Eredeti fotó", (8,8))


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
    img = cv.resize(image_bgr, (target_width, int(image_bgr.shape[0]*ratio)))
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray=cv.GaussianBlur(gray,(5,5),0)
    edges=cv.Canny(gray,60,180)
    edges=cv.dilate(edges,np.ones((3,3),np.uint8),1)
    quad=largest_quad_contour(edges)
    if quad is None:
        print("Nem találtam egyértelmű papírlap-kontúrt, marad a teljes kép.")
        return img, False
    pts=order_pts(quad)
    w=int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    h=int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    M=cv.getPerspectiveTransform(pts, np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32))
    warped=cv.warpPerspective(img, M, (w,h))
    return warped, True

raw, ok = warp_document(img)
show(raw, "RAW (kivágva + kiegyenesítve)", (8,8))


def enhance_for_ocr(bgr):
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # finom denoise – kevesebb szemcse
    gray = cv.GaussianBlur(gray, (5,5), 0)
    # globális Otsu a mostani textúrára jobb, mint az adaptív
    _, binimg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # apró pöttyök eltávolítása
    binimg = cv.morphologyEx(binimg, cv.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return gray, binimg

gray, binimg = enhance_for_ocr(raw)
show(gray, "PROC – kontraszt (szürke)")
show(binimg, "PROC – binarizált")


reader = easyocr.Reader(['en'], gpu=False)

def ocr_easy(img_gray):
    res = reader.readtext(
        img_gray,
        detail=0,
        paragraph=True,
        decoder='beamsearch',     # pontosabb, de kicsit lassabb
        text_threshold=0.4,       # jobban felismeri a halvány betűket
        low_text=0.3
    )
    return "\n".join(res)

def ocr_tess(img_gray):
    cfg = "--oem 1 --psm 4 -l eng"
    return pytesseract.image_to_string(img_gray, config=cfg)

txt_raw_ez  = ocr_easy(cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
txt_proc_ez = ocr_easy(binimg)
txt_raw_te  = ocr_tess(cv.cvtColor(raw, cv.COLOR_BGR2GRAY))
txt_proc_te = ocr_tess(binimg)

print("=== EasyOCR RAW ===\n", txt_raw_ez)
print("\n=== EasyOCR PROC ===\n", txt_proc_ez)
print("\n=== Tesseract RAW ===\n", txt_raw_te)
print("\n=== Tesseract PROC ===\n", txt_proc_te)

# --- Várt szöveg bekérése ---
expected = input("\nÍrd be, milyen szöveget írtál a papírra (elvárt szöveg, ékezet nélkül):\n")

def sim(a,b):
    return difflib.SequenceMatcher(None,a.upper().strip(),b.upper().strip()).ratio()

print("\nElvárt szöveg:")
print(expected, "\n")

for name, txt in [("Easy RAW",txt_raw_ez),
                  ("Easy PROC",txt_proc_ez),
                  ("Tess RAW",txt_raw_te),
                  ("Tess PROC",txt_proc_te)]:
    print(f"{name:10s} -> match={sim(expected, txt):.2f}")

# --- Mentés outputs mappába (projekt gyökerében) ---
os.makedirs("outputs", exist_ok=True)
cv.imwrite("outputs/raw.jpg", raw)
cv.imwrite("outputs/proc_bin.png", binimg)
with open("outputs/ocr.json","w",encoding="utf-8") as f:
    json.dump({"easy_raw":txt_raw_ez,"easy_proc":txt_proc_ez,
               "tess_raw":txt_raw_te,"tess_proc":txt_proc_te},
              f, ensure_ascii=False, indent=2)
print("\nMentve: outputs/")
