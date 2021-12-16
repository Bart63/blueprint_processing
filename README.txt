'median_blur' - liczba dodatnia nieparzysta. Rozmycie przed usuwaniem najmniej wyraźnych pikseli - głównie mebli.

'gauss_blur' - liczba dodatnia nieparzysta. Rozmycie przed przekształceniem binarnym. Większe wartości poszerzą obszerniejsze elementy i zmniejszą intensywność mniejszych.

'threshold' - całkowita od 0 do 255. Zerujemy piksele poniżej granicy. Pozostałe ustawiamy na biały. Im mniejsza, tym mniej śmieci, ale też potencjalne cieńsze ściany możemy usunąć.

'poly_approx' - ułamkowa wartość aproksymująca znaleziony kształt do wielokąta. Im mniejsza, tym dokładniejsze odwzorowanie - więcej wierzchołków.

'min_area' - dodatnia liczba całkowita. Minimalna powierzchnia odkrytego wielokąt w pikselach.

'max_area_percent' - dodatni ułamek z zakresu od 0 do 1. Maksymalna powierzchnia odkrytego wielokąta wyrażana w procentach. Procent powierzchni zdjęcia planu.

'remove_near_vertex' - dodatni dystans w pikselach. Wartość do iterującego usuwania najbliższych pikseli w każdym z wykrytych wielokątów.

'linearize_offset' - dodatni dystans w pikselach. Dla każdych dwóch kolejnych wierzchołków sprawdzana jest różnica odległości w dwóch wymiarach. Jeżeli wartość jest mniejsza niż podana, to wyrównujemy drugi wierzchołek w tym wymiarze.

