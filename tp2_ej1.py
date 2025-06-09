import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
os.environ['OMP_NUM_THREADS'] = '1' # Para evitar un warning al hacer kmeans contando los capacitores

# =================== RESISTENCIAS ====================
def detectar_resistencias(img, mostrar = False):
    
    img_resistencia = img.copy()
    # --- Preprocesamiento ---
    gray = cv2.cvtColor(img_resistencia, cv2.COLOR_BGR2GRAY)

    # --- Umbral automático con "THRESH_OTSU" ---
    th_out, thresh = cv2.threshold(gray, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Cierre morfológico ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # --- Apertura morfológica ---
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    apertura = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel2)

    # --- Detecta contornos ----
    contornos, _ = cv2.findContours(apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resistencias = 0 # inicio contador de resistencias en 0

    for c in contornos:  # saltamos el fondo
        x, y, w, h = cv2.boundingRect(c)      # Recupero las coordenadas del contorno
        label = "Resistencia"
        color = (0, 255, 0)  # Verde
        aspect_ratio = w / h if h != 0 else 0
        area = w * h

        if (7000 < area)and((aspect_ratio >= 2.0) or (aspect_ratio <= 0.35)):

            label = "Resistencia"
            color = (0, 255, 0)  # Verde
            resistencias += 1

            # Dibujar caja y etiqueta
            cv2.rectangle(img_resistencia, (x, y), (x + w, y + h), color, 4)
            cv2.putText(img_resistencia, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)      

    # --- Gráficos auxiliares del proceso --- 
    if mostrar == True:
        # --- Mostrar resultados ---
        titles = ['Gray', 'Adaptive Threshold', 'Closed', 'Opened', 'Output']
        images = [gray, thresh, closed, apertura, img_resistencia]
        plt.figure(figsize=(15, 7))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            cmap = 'gray' if i < 4 else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # --- Guardar la imagen ---
    cv2.imwrite('Imagenes/resistencias_detectadas.png', img_resistencia)

    # --- Gráfico con resistencias detectadas ---

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_resistencia, cv2.COLOR_BGR2RGB))
    plt.title(f'Resistencias: {resistencias}')
    plt.axis('off')
    plt.show()

    print('==========================================')
    print(f'Resistencias detectadas: {resistencias}')

# ==================== CHIP ===========================
def detectar_chip(img, mostrar = False):

    img_chip = img.copy()
    gray = cv2.cvtColor(img_chip, cv2.COLOR_BGR2GRAY)
    # --- Adaptive Threshold (mejor para iluminación desigual) ---
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        23, 5  # Ajustables
    )
    # --- Cierre morfológico ---
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    # --- Dilatar ---
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(closed, kernel_dilate, iterations=1)
    # --- Componentes conexas ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    
    chip = 0 # inicio contador de chip en 0, si bien hay uno solo

    # --- Clasificación por tamaño y por relación de aspecto ---
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = w / h if h != 0 else 0
        
        if (60000 < area < 61000) and (0.7 >= aspect_ratio >= 0.4):
            label = "Chip"
            color = (0, 255, 0)  # Verde
            chip += 1
            # Dibujar caja y etiqueta
            cv2.rectangle(img_chip, (x, y), (x + w, y + h), color, 5)
            cv2.putText(img_chip, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    # --- Mostrar resultados ---
    if mostrar == True:
        titles = ['Gray', 'Adaptive Threshold', 'Closed', 'Dilated', 'Output']
        images = [gray, thresh, closed, dilated, img_chip]

        # --- Gráficos auxiliares del proceso --- 
        plt.figure(figsize=(15, 7))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            cmap = 'gray' if i < 4 else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # --- Guardar la imagen ---
    cv2.imwrite('Imagenes/chip_detectado.png', img_chip)

    # --- Gráfico con el chip detectado ---

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_chip, cv2.COLOR_BGR2RGB))
    plt.title(f'Chip: {chip}')
    plt.axis('off')
    plt.show()

    print('==========================================')
    print(f'Chip: {chip}')

# ==================== CAPACITORES ====================
def detectar_capacitores(img, mostrar = False):

    # --- Crear imagen blanca del mismo tamaño que la original ---
    img_capacitores = img.copy()
    gray = cv2.cvtColor(img_capacitores, cv2.COLOR_BGR2GRAY)

    # --- Convertir a formato requerido para HoughCircles ---
    gray_blur = cv2.medianBlur(gray, 5)  # reduce ruido salt

    # --- Detectar círculos ---
    circles = cv2.HoughCircles(
        gray_blur, 
        cv2.HOUGH_GRADIENT,
        dp=1.2,            # resolución del acumulador
        minDist=200,       # distancia mínima entre centros de círculos
        param1=100,        # umbral alto de Canny interno
        param2=115,        # umbral acumulador (más alto = más selectivo)
        minRadius=25,      # RADIO mínimo
        maxRadius=300      # RADIO máximo
    )

    capacitores = 0

    # --- Dibujar círculos detectados ---
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            capacitores += 1
            # círculo en azul
            cv2.circle(img_capacitores, center, radius, (0, 255, 0), 5)
            # centro en rojo
            cv2.circle(img_capacitores, center, 4, (0, 0, 255), 5)
        print('==========================================')
        print(f'Cantidad de capacitores: {len(circles[0])}')
    else:
        print('No se detectaron círculos.')

    # --- Mostrar resultados ---
    if mostrar == True:
        titles = ['Gray', 'Gray Blur', 'Output']
        images = [gray, gray_blur, img_capacitores]

        # --- Gráficos auxiliares del proceso ---
        plt.figure(figsize=(14, 7))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            cmap = 'gray' if i < 4 else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # --- Guardar la imagen ---
    cv2.imwrite('Imagenes/capacitores_detectados.png', img_capacitores)

    # --- Gráfico con los capacitores detectados ---

    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(img_capacitores, cv2.COLOR_BGR2RGB))
    plt.title(f'Capacitores: {capacitores}')
    plt.axis('off')
    plt.show()

    # # --- Conteo de capacitores por tamaño ---
    # # --- Forma manual por el tamaño del círculo ---
    # grupo_1 = 0  # Pequeños
    # grupo_2 = 0  # Medianos
    # grupo_3 = 0  # Grandes
    # grupo_4 = 0  # Muy grandes

    # # --- Divide según el radio
    # if circles is not None:
    #     for c in circles[0]:
    #         radio = c[2]
    #         if 68 < radio < 76:
    #             grupo_1 += 1
    #         elif 91 < radio < 95:
    #             grupo_2 += 1
    #         elif 146 < radio < 153:
    #             grupo_3 += 1
    #         elif 245 < radio:
    #             grupo_4 += 1

    #     print("Capacitores clasificados manualmente:")
    #     print(f"- Grupo 1 (pequeños): {grupo_1}")
    #     print(f"- Grupo 2 (medianos): {grupo_2}")
    #     print(f"- Grupo 3 (grandes): {grupo_3}")
    #     print(f"- Grupo 4 (muy grandes): {grupo_4}")
    # else:
    #     print("No se detectaron capacitores.")

    # --- Forma más genérica y abarcativa ---
    if circles is not None:
        # --- Extrae radios y aplica KMeans con 4 grupos ---
        radios = np.array([c[2] for c in circles[0]]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(radios)
        etiquetas = kmeans.labels_

        # --- Ordena en clusters por tamaño de radio (de menor a mayor) ---
        orden_clusters = np.argsort(kmeans.cluster_centers_.reshape(-1))

        # --- Cuenta los elementos en cada grupo ordenado ---
        conteo = [np.sum(etiquetas == i) for i in orden_clusters]

        # --- Imprimir resultados ---
        print('==========================================')
        print("Capacitores clasificados por clustering:")
        print("(ordenados de menor a mayor)")
        for idx, cantidad in enumerate(conteo, 1):
            print(f"- Grupo {idx}: {cantidad}")
    else:
        print("No se detectaron capacitores.")

# =================== PROCEDIMIENTO ===================
img = cv2.imread('Imagenes/placa.png') # cargo imagen para pasar por parametros
# Para mostrar imagenes del procesamiento de la imagen, mostrar = True
detectar_resistencias(img, mostrar=False)
detectar_chip(img, mostrar=False)
detectar_capacitores(img, mostrar=False)
