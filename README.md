# OCR Comparison – TDK Project

Szövegfelismerő algoritmusok összehasonlító vizsgálata valós környezetben készült dokumentumképeken

A projekt saját készítésű (telefonos) és nyilvános dataset képeken vizsgálja az OCR felismerés pontosságát.

---

## Projekt célja

A pipeline egy lefotózott papírlap automatikus feldolgozását végzi:

1. Dokumentum detektálása a képen
2. Perspektívakorrekció
3. Zajszűrés és kontrasztjavítás
4. Binarizálás
5. OCR szövegfelismerés
6. Felismert szöveg összehasonlítása referencia szöveggel
7. Eredmények mentése

A cél különböző OCR megoldások teljesítményének mérése.

---

## Jelenlegi megvalósítás

A program:

- képet tölt be fájlból
- dokumentum kontúrt keres
- kiegyenesíti a dokumentumot
- előfeldolgozza OCR számára
- OCR felismerést futtat
- összehasonlítja az eredményt a referencia szöveggel
- elmenti a feldolgozott képet és az OCR eredményeket

---

## Használt technológiák

- Python 3
- OpenCV
- EasyOCR
- Tesseract OCR
- PaddleOCR
- KerasOCR (opcionális)
- docTR (opcionális)
- NumPy
- Matplotlib

---

## Futtatás

Telepítsd a szükséges csomagokat:

```bash
pip install opencv-python-headless easyocr pytesseract matplotlib numpy
