import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def wczytaj_zdjecie(url):
    try:
        headery = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        odpowiedz = requests.get(url, headers=headery, timeout=15)
        odpowiedz.raise_for_status()
        obraz = Image.open(BytesIO(odpowiedz.content))
        return obraz.convert('RGB')
    except Exception as e:
        print(f"Error podczas wczytywania zdjecia: {e}")
        return None

def oblicz_histogram(obraz):
    tablica_obrazu = np.array(obraz)
    
    hist_c = np.histogram(tablica_obrazu[:,:,0], bins=256, range=(0, 256))[0]
    hist_z = np.histogram(tablica_obrazu[:,:,1], bins=256, range=(0, 256))[0]
    hist_n = np.histogram(tablica_obrazu[:,:,2], bins=256, range=(0, 256))[0]
    hist_calkowity = hist_c + hist_z + hist_n
    
    return hist_c, hist_z, hist_n, hist_calkowity

def wyswietl_histogramy(hist_c, hist_z, hist_n, hist_calkowity):
    rys, osie = plt.subplots(2, 2, figsize=(12, 10))
    
    osie[0, 0].plot(hist_calkowity, color='black')
    osie[0, 0].set_title('Histogram calkowity')
    osie[0, 0].set_xlabel('Wartosc piksela')
    osie[0, 0].set_ylabel('Liczba pikseli')
    osie[0, 0].grid(True, alpha=0.3)
    
    osie[0, 1].plot(hist_c, color='red')
    osie[0, 1].set_title('Kanal czerwony')
    osie[0, 1].set_xlabel('Wartosc piksela')
    osie[0, 1].set_ylabel('Liczba pikseli')
    osie[0, 1].grid(True, alpha=0.3)
    
    osie[1, 0].plot(hist_z, color='green')
    osie[1, 0].set_title('Kanal zielony')
    osie[1, 0].set_xlabel('Wartosc piksela')
    osie[1, 0].set_ylabel('Liczba pikseli')
    osie[1, 0].grid(True, alpha=0.3)
    
    osie[1, 1].plot(hist_n, color='blue')
    osie[1, 1].set_title('Kanal niebieski')
    osie[1, 1].set_xlabel('Wartosc piksela')
    osie[1, 1].set_ylabel('Liczba pikseli')
    osie[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def ocen_jakosc_zdjecia(hist_calkowity, tablica_obrazu):
    calkowita_liczba_pikseli = np.sum(hist_calkowity)
    
    wartosci_pikseli = np.repeat(np.arange(256), hist_calkowity.astype(int))
    odch_std = np.std(wartosci_pikseli)
    ocena_kontrastu = min(odch_std / 50.0, 1.0)
    
    niezerowe = np.where(hist_calkowity > 0)[0]
    if len(niezerowe) > 0:
        zakres_wartosci = niezerowe[-1] - niezerowe[0]
        ocena_zakresu_dynamicznego = zakres_wartosci / 255.0
    else:
        zakres_wartosci = 0
        ocena_zakresu_dynamicznego = 0.0
    
    srednia_wartosc = np.mean(wartosci_pikseli)
    if srednia_wartosc < 85:
        status_ekspozycji = "niedoswietlone"
        ocena_ekspozycji = srednia_wartosc / 85.0
    elif srednia_wartosc > 170:
        status_ekspozycji = "przeswietlone"
        ocena_ekspozycji = (255 - srednia_wartosc) / 85.0
    else:
        status_ekspozycji = "prawidlowe"
        ocena_ekspozycji = 1.0
    
    hist_znormalizowany = hist_calkowity / calkowita_liczba_pikseli
    hist_znormalizowany = hist_znormalizowany[hist_znormalizowany > 0]
    entropia = -np.sum(hist_znormalizowany * np.log2(hist_znormalizowany))
    ocena_entropii = min(entropia / 8.0, 1.0)
    
    obcinanie_niskie = hist_calkowity[0] / calkowita_liczba_pikseli
    obcinanie_wysokie = hist_calkowity[255] / calkowita_liczba_pikseli
    ocena_obcinania = 1.0 - max(obcinanie_niskie, obcinanie_wysokie) * 10
    ocena_obcinania = max(0.0, ocena_obcinania)
    
    jakosc = (
        ocena_kontrastu * 0.25 +
        ocena_zakresu_dynamicznego * 0.25 +
        ocena_ekspozycji * 0.20 +
        ocena_entropii * 0.20 +
        ocena_obcinania * 0.10
    ) * 100
    
    print("RAPORT JAKOSCI ZDJĘCIA")
    print(f"\nKontrast: {ocena_kontrastu*100:.1f}% (std. dev: {odch_std:.2f})")
    print(f"Zakres dynamiczny: {ocena_zakresu_dynamicznego*100:.1f}% ({zakres_wartosci}/255 poziomów)")
    print(f"Ekspozycja: {ocena_ekspozycji*100:.1f}% - {status_ekspozycji} (średnia: {srednia_wartosc:.1f})")
    print(f"Entropia: {ocena_entropii*100:.1f}% (wartość: {entropia:.2f})")
    print(f"Obcinanie histogramu: {ocena_obcinania*100:.1f}% \n")
    
    if obcinanie_niskie > 0.01:
        print(f"   - Ostrzezenie: {obcinanie_niskie*100:.2f}% pikseli w czerni (0)")
    if obcinanie_wysokie > 0.01:
        print(f"   - Ostrzezenie: {obcinanie_wysokie*100:.2f}% pikseli w bieli (255)")
    
    print(f"OGOLNA OCENA JAKOSCI: {jakosc:.1f}/100")
    
    if jakosc >= 80:
        poziom_jakosci = "WYSOKA"
    elif jakosc >= 60:
        poziom_jakosci = "DOBRA"
    elif jakosc >= 40:
        poziom_jakosci = "ŚREDNIA"
    else:
        poziom_jakosci = "NISKA"
    
    print(f"\nJakosc zdjecia: {poziom_jakosci}\n")
    
    return jakosc

def main():
    url = "https://upload.wikimedia.org/wikipedia/commons/6/68/Eurasian_wolf_2.jpg"
    
    print(f"Wczytuje zdjecie z: {url} \n")
    
    obraz = wczytaj_zdjecie(url)
    if obraz is None:
        print("Nie udalo się wczytac zdjecia.")
        return
    
    print(f"Zdjecie wczytane Rozmiar: {obraz.size} \n")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(obraz)
    plt.title('Analizowane zdjecie')
    plt.axis('off')
    plt.show()
    
    hist_c, hist_z, hist_n, hist_calkowity = oblicz_histogram(obraz)
    
    wyswietl_histogramy(hist_c, hist_z, hist_n, hist_calkowity)

    tablica_obrazu = np.array(obraz)
    ocena_jakosci = ocen_jakosc_zdjecia(hist_calkowity, tablica_obrazu)

if __name__ == "__main__":
    main()
