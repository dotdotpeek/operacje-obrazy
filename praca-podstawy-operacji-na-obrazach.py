import urllib.request
import numpy as np
import cv2
import matplotlib.pyplot as plt

print("Pobieranie obrazu")
url = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Citrons_NICE_2003.jpg"

req = urllib.request.Request(
    url,
    headers={'User-Agent': 'Mozilla/5.0'}
)

resp = urllib.request.urlopen(req)
data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
print("Obraz pobrany.")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Oryginalny obraz")
plt.axis("off")
plt.show()
print("\n2. Zmniejszanie rozdzielczości o 50")
h, w = img.shape[:2]
new_w = w // 2
new_h = h // 2
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
print(f"   Nowe wymiary: {new_w} x {new_h}")
print("\n3. Konwersja do skali szarości...")
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
print("Konwersja zakończona. Wymiary macierzy:", gray.shape)
print("\n4. Obracanie obrazu o 90 stopni")
obrocony = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
print(f"   Wymiary po obrocie: {obrocony.shape[1]} x {obrocony.shape[0]}")
plt.imshow(obrocony, cmap="gray")
plt.title("Obraz po przetwarzaniu")
plt.axis("off")
plt.show()
print("\n6. Macierz pikseli obrazu po przetwarzaniu:")
print(obrocony)