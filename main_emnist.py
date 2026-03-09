# ============================================
# IMPORT
# ============================================

import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import easyocr
import random
import difflib
import tempfile
import os

from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from torchvision.datasets import EMNIST
from torchvision import transforms

random.seed(42)

print("Torch verzió:", torch.__version__)

# ============================================
# OCR MODELLEK BETÖLTÉSE
# ============================================

reader = easyocr.Reader(['en'], gpu=False)
doctr_model = ocr_predictor(pretrained=True)

print("OCR modellek betöltve")

# Ha Windows alatt szükséges:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================
# EMNIST BETÖLTÉS
# ============================================

transform = transforms.Compose([transforms.ToTensor()])

dataset = EMNIST(
    root="./emnist_data",
    split="letters",
    train=True,
    download=True,
    transform=transform
)

print("EMNIST méret:", len(dataset))

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ============================================
# SEGÉDFÜGGVÉNYEK
# ============================================

def normalize(text):
    text = text.replace("\n","")
    text = text.replace(" ","")
    text = text.replace("\x0c","")
    return text.upper().strip()


def similarity(a,b):
    return difflib.SequenceMatcher(None,a,b).ratio()


def rotate_emnist(img):
    img = np.rot90(img,3)
    img = np.fliplr(img)
    return img


def crop(img):
    ys,xs = np.where(img>20)

    if len(xs)==0:
        return img

    x1,x2 = xs.min(), xs.max()
    y1,y2 = ys.min(), ys.max()

    return img[y1:y2+1, x1:x2+1]

# ============================================
# KARAKTER KÉP GENERÁLÁS
# ============================================

def make_char(img):

    img = rotate_emnist(img)
    img = (img*255).astype(np.uint8)

    img = 255-img

    img = crop(img)

    img = cv.resize(img,(120,180),interpolation=cv.INTER_CUBIC)

    _,img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    inv = 255-img
    kernel = np.ones((3,3),np.uint8)

    inv = cv.dilate(inv,kernel,1)

    img = 255-inv

    return img


# ============================================
# KARAKTERBANK
# ============================================

print("Karakterbank építése...")

char_bank = {l:[] for l in LETTERS}

for img,label in dataset:

    letter = LETTERS[(int(label)-1)%26]

    if len(char_bank[letter])>=80:
        continue

    arr = img.numpy()[0]

    arr = make_char(arr)

    char_bank[letter].append(arr)

    if all(len(v)>=80 for v in char_bank.values()):
        break

print("Karakterbank kész")


# ============================================
# RANDOM SZÓ
# ============================================

def random_word():

    length = random.randint(4,7)

    return "".join(random.choice(LETTERS) for _ in range(length))


# ============================================
# SZÓ KÉP GENERÁLÁS
# ============================================

def make_word(word):

    parts=[]

    for ch in word:

        img=random.choice(char_bank[ch])

        parts.append(img)

        parts.append(np.ones((180,40),dtype=np.uint8)*255)

    line=np.hstack(parts[:-1])

    canvas=cv.copyMakeBorder(
        line,
        80,80,100,100,
        cv.BORDER_CONSTANT,
        value=255
    )

    canvas=cv.GaussianBlur(canvas,(3,3),0)

    return canvas


# ============================================
# BINARIZÁLÁS
# ============================================

def preprocess(img):

    gray=img.copy()

    binimg=cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        31,
        10
    )

    inv=255-binimg

    kernel=np.ones((2,2),np.uint8)

    inv=cv.dilate(inv,kernel,1)

    binimg=255-inv

    bgr=cv.cvtColor(binimg,cv.COLOR_GRAY2BGR)

    return gray,binimg,bgr


# ============================================
# OCR
# ============================================

def ocr_easy(img):

    res=reader.readtext(
        img,
        detail=0,
        paragraph=True,
        decoder='beamsearch',
        allowlist=LETTERS
    )

    return "".join(res)


def ocr_tess(img):

    cfg="--oem 1 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    return pytesseract.image_to_string(img,config=cfg)


def ocr_doctr(img):

    path=os.path.join(tempfile.gettempdir(),"doctr.jpg")

    cv.imwrite(path,img)

    doc=DocumentFile.from_images(path)

    res=doctr_model(doc)

    words=[]

    for page in res.pages:
        for block in page.blocks:
            for line in block.lines:
                for w in line.words:
                    words.append(w.value)

    return "".join(words)


# ============================================
# TESZT + MENTÉS
# ============================================

N = 10

output_dir = "emnist_results"
os.makedirs(output_dir, exist_ok=True)

results = []

acc_easy=[]
acc_tess=[]
acc_doctr=[]

for i in range(N):

    expected=random_word()

    img=make_word(expected)

    gray,binimg,bgr=preprocess(img)

    easy=normalize(ocr_easy(binimg))
    tess=normalize(ocr_tess(binimg))
    doctr=normalize(ocr_doctr(bgr))

    exp=normalize(expected)

    sim_easy=similarity(exp,easy)
    sim_tess=similarity(exp,tess)
    sim_doctr=similarity(exp,doctr)

    acc_easy.append(sim_easy)
    acc_tess.append(sim_tess)
    acc_doctr.append(sim_doctr)

    # ----------------------------------------
    # KÉP MENTÉS
    # ----------------------------------------

    img_name=f"sample_{i:03d}.png"

    img_path=os.path.join(output_dir,img_name)

    cv.imwrite(img_path,gray)

    # ----------------------------------------
    # EREDMÉNY LISTA
    # ----------------------------------------

    results.append({
        "id":i,
        "expected":exp,
        "easyocr":easy,
        "tesseract":tess,
        "doctr":doctr,
        "sim_easy":sim_easy,
        "sim_tess":sim_tess,
        "sim_doctr":sim_doctr,
        "image":img_name
    })

    # ----------------------------------------
    # KIÍRÁS
    # ----------------------------------------

    print("\n==============================")
    print("ID:",i)
    print("ELVÁRT:",exp)
    print("EasyOCR :",easy)
    print("Tesseract:",tess)
    print("docTR :",doctr)

    # ----------------------------------------
    # KÉP MEGJELENÍTÉS
    # ----------------------------------------

    plt.figure(figsize=(6,3))
    plt.imshow(gray,cmap="gray")
    plt.title(f"{i} | expected: {exp}")
    plt.axis("off")
    plt.show()


# ============================================
# ÁTLAG PONTOSSÁG
# ============================================

print("\nEMNIST SZÓ TESZT")
print("========================================")

print("EasyOCR   :",round(np.mean(acc_easy)*100,2),"%")
print("Tesseract :",round(np.mean(acc_tess)*100,2),"%")
print("docTR     :",round(np.mean(acc_doctr)*100,2),"%")


# ============================================
# CSV MENTÉS
# ============================================

import csv

csv_path=os.path.join(output_dir,"results.csv")

with open(csv_path,"w",newline="",encoding="utf-8") as f:

    writer=csv.writer(f)

    writer.writerow([
        "id",
        "expected",
        "easyocr",
        "tesseract",
        "doctr",
        "sim_easy",
        "sim_tess",
        "sim_doctr",
        "image"
    ])

    for r in results:

        writer.writerow([
            r["id"],
            r["expected"],
            r["easyocr"],
            r["tesseract"],
            r["doctr"],
            r["sim_easy"],
            r["sim_tess"],
            r["sim_doctr"],
            r["image"]
        ])

print("\nMentve ide:")
print(csv_path)
