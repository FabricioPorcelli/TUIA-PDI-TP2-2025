import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import re

# vars globales para mostrar los graficos
mostrar_graficos_1 = False # etapa 1
mostrar_graficos_2 = False # etapa 2

''' ===============================================================================================
    ============== PRIMERA ETAPA: Recorte de rectangulos azules con las resistencias ============== '''

# Función para obtener una máscara de las áreas de color azul fuerte en una imagen
def obtener_mascara_azul_fuerte(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # Convierte la imagen de BGR a HSV
    lower_blue = np.array([90, 50, 50])  # Límite inferior para el rango de color azul
    upper_blue = np.array([130, 255, 255])  # Límite superior para el rango de color azul
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)  # Genera una máscara para los píxeles en el rango

    # Graficos demostrativos del proceso
    if mostrar_graficos_1 == True:
        plt.figure(figsize=(15, 7))
        plt.imshow(mask)
        plt.title('Mascara azul')
        plt.axis('off')
        plt.show() 
    
    return mask

# Función para ordenar los puntos de un contorno en un orden específico (superior izq -> sup der -> inf der -> inf izq)
def ordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")  # Inicializa una matriz de 4 puntos
    s = puntos.sum(axis=1)  # Suma de las coordenadas x + y
    diff = np.diff(puntos, axis=1)  # Diferencia entre las coordenadas x - y

    rect[0] = puntos[np.argmin(s)]     # Punto con la menor suma (superior izquierdo)
    rect[2] = puntos[np.argmax(s)]     # Punto con la mayor suma (inferior derecho)
    rect[1] = puntos[np.argmin(diff)]  # Menor diferencia (superior derecho)
    rect[3] = puntos[np.argmax(diff)]  # Mayor diferencia (inferior izquierdo)

    return rect

# Función híbrida para obtener una vista en perspectiva desde arriba de una imagen con una región azul
def perspectiva_vista_superior_hibrido(img_bgr):
    # Sub-función que intenta detectar un contorno rectangular preciso
    def metodo_contorno_preciso(img_bgr):
        mask = obtener_mascara_azul_fuerte(img_bgr)  # Obtiene la máscara azul fuerte
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Kernel rectangular para operaciones morfológicas
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Aplica cierre para unir regiones cercanas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Encuentra contornos

        mejor_contorno = None  # Inicializa la variable para almacenar el mejor contorno
        max_area = 0  # Área máxima inicial

        # Recorre cada contorno para encontrar el mejor candidato
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)  # Perímetro del contorno
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # Aproxima el contorno a un polígono

            # Verifica si el contorno tiene 4 lados y mayor área.
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    mejor_contorno = approx

        if mejor_contorno is None:
            raise ValueError("Contorno rectangular no detectado.")  # Lanza excepción si no encuentra un contorno válido

        box_ordenado = ordenar_puntos(mejor_contorno.reshape(4, 2))  # Ordena los puntos del contorno
        return transformar_perspectiva(img_bgr, box_ordenado)  # Realiza la transformación en perspectiva

    # Sub-función alternativa que usa un rectángulo de área mínima si el contorno preciso falla
    def metodo_fallback_min_area_rect(img_bgr):
        mask = obtener_mascara_azul_fuerte(img_bgr)  # Obtiene la máscara azul fuerte
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Kernel más pequeño para el cierre
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Aplica cierre para unir regiones cercanas
        coords = cv2.findNonZero(mask)  # Obtiene las coordenadas no nulas de la máscara

        if coords is None:
            raise ValueError("No se detectó región azul.")  # Lanza excepción si no hay píxeles detectados

        rect = cv2.minAreaRect(coords)  # Calcula el rectángulo de área mínima que encierra las coordenadas
        box = cv2.boxPoints(rect)  # Convierte el rectángulo en un conjunto de 4 puntos
        box = box.astype(np.intp)  # Convierte las coordenadas a enteros
        box_ordenado = ordenar_puntos(box)  # Ordena los puntos

        return transformar_perspectiva(img_bgr, box_ordenado)  # Realiza la transformación en perspectiva

    # Sub-función para transformar una región de interés en perspectiva
    def transformar_perspectiva(img, box_ordenado):
        (tl, tr, br, bl) = box_ordenado  # Extrae los puntos en el orden deseado
        # Calcula las dimensiones del nuevo plano
        ancho_sup = np.linalg.norm(tr - tl)
        ancho_inf = np.linalg.norm(br - bl)
        alto_izq = np.linalg.norm(bl - tl)
        alto_der = np.linalg.norm(br - tr)
        ancho_max = int(max(ancho_sup, ancho_inf))
        alto_max = int(max(alto_izq, alto_der))

        # Define los puntos destino en el nuevo plano
        destino = np.array([
            [0, 0],
            [ancho_max - 1, 0],
            [ancho_max - 1, alto_max - 1],
            [0, alto_max - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(box_ordenado, destino)  # Calcula la matriz de transformación

        transf = cv2.warpPerspective(img, M, (ancho_max, alto_max))

        # Gráficos auxiliares del proceso 
        if mostrar_graficos_1 == True:
            titles = ['Original', 'Transformada']
            images = [img, transf]
            plt.figure(figsize=(15, 7))
            for i in range(2):
                plt.subplot(1, 2, i+1)
                cmap = 'gray' if i < 4 else None
                plt.imshow(images[i], cmap=cmap)
                plt.title(titles[i])
                plt.axis('off')
            plt.tight_layout()
            plt.show()

        return transf  # Aplica la transformación

    # Intenta usar el método confiable; si falla, usa el método de respaldo
    try:
        return metodo_contorno_preciso(img_bgr)
    except Exception as e:
        return metodo_fallback_min_area_rect(img_bgr)

''' ===============================================================================================
    =========== SEGUNDA ETAPA: Detección del cuerpo y bandas de color de la resistencia =========== '''

# Detecta y segmenta el cuerpo de la resistencia en una imagen con fondo azul.
# Devuelve:
#  - la región recortada (ROI) del cuerpo de la resistencia.
def detectar_cuerpo_resistencia(imagen):
    # Convertir a espacio de color HSV
    img_hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    output = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Define el rango HSV para detectar el cuerpo de la resistencia
    umbral_bajo = np.array([20, 50, 50])  # Ajustar valores según sea necesario
    umbral_alto = np.array([150, 255, 255])
    mascara_umbrales = cv2.inRange(img_hsv, umbral_bajo, umbral_alto)  # Máscara binaria

    # Operación de cierre para rellenar espacios pequeños
    kernel_clausura = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 27))
    mascara_clausura = cv2.morphologyEx(mascara_umbrales, cv2.MORPH_CLOSE, kernel_clausura)

    # Operación de apertura para eliminar ruido
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mascara_apertura = cv2.morphologyEx(mascara_clausura, cv2.MORPH_OPEN, kernel_apertura)

    # Detectar contornos
    contornos, _ = cv2.findContours(mascara_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar ROI vacío
    roi = None

    for c in contornos:
        # Obtener el rectángulo delimitador del contorno más grande
        x, y, w, h_c = cv2.boundingRect(c)
        if w > 30 and h_c > 10:  # Filtrar por tamaño mínimo
            # Dibujar el rectángulo en la imagen de salida
            cv2.rectangle(output, (x, y), (x + w, y + h_c), (255, 0, 0), 2)
            cv2.putText(output, 'Resistencia', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # Recortar la región de interés (ROI)
            roi = imagen[y:y + h_c, x:x + w]
            break  # Asumimos que solo hay una resistencia por imagen

    # --- Gráficos auxiliares del proceso --- 
    if mostrar_graficos_2 == True:
        # --- Mostrar resultados ---
        titles = ['mascara_umbrales', 'mascara_clausura', 'mascara_apertura', 'output', 'roi']
        images = [mascara_umbrales, mascara_clausura, mascara_apertura, output, roi]
        plt.figure(figsize=(15, 7))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            cmap = 'gray' if i < 4 else None
            plt.imshow(images[i], cmap=cmap)
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Graficos demostrativos del proceso
    if mostrar_graficos_2 == True:
        plt.figure(figsize=(15, 7))
        plt.imshow(output)
        plt.title('Cuerpo de la resistencia detectado -> ROI')
        plt.axis('off')
        plt.show()    
    
    return roi, output

# Detecta las bandas de color en una imagen HSV.
# Devuelve:
#  - img_con_bandas (numpy.ndarray): Imagen con las bandas de color marcadas.
#  - bandas_detectadas (list): Lista de coordenadas x y nombres de colores detectados.
def detectar_colores_bandas(img_hsv):

    # Convertir la imagen a espacio RGB para mostrar resultados
    img_con_bandas2 = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_con_bandas = cv2.bilateralFilter(img_con_bandas2, d=9, sigmaColor=75, sigmaSpace=75)

    # Definir rangos HSV aproximados para los colores de las bandas
    rangos_colores = [
        {"color": "Negro", "bajo": np.array([0, 0, 0]), "alto": np.array([179, 255, 40])},
        {"color": "Marron", "bajo": np.array([0, 130, 0]), "alto": np.array([10, 255, 120])},
        {"color": "Rojo", "bajo": np.array([0, 140, 100]), "alto": np.array([10, 255, 255])},
        {"color": "Naranja", "bajo": np.array([10, 160, 150]), "alto": np.array([15, 255, 255])},
        {"color": "Amarillo", "bajo": np.array([20, 140, 0]), "alto": np.array([35, 255, 255])},
        {"color": "Verde", "bajo": np.array([35, 50, 50]), "alto": np.array([75, 255, 255])},
        {"color": "Violeta", "bajo": np.array([130, 80, 70]), "alto": np.array([179, 255, 255])},
        {"color": "Blanco", "bajo": np.array([0, 35, 90]), "alto": np.array([15, 90, 190])},
    ]

    bandas_detectadas = []

    # Iterar sobre cada rango de color
    for rango in rangos_colores:
        # Crear una máscara para el color actual
        mascara = cv2.inRange(img_hsv, rango["bajo"], rango["alto"])
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mascara_procesada = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

        # Detectar contornos en la máscara procesada
        contornos, _ = cv2.findContours(mascara_procesada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Procesar cada contorno
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 600:  # Filtrar contornos pequeños
                x, y, w, h = cv2.boundingRect(contorno)
                bandas_detectadas.append((x, rango["color"]))

                # Dibujar el rectángulo y el nombre del color en la imagen
                cv2.rectangle(img_con_bandas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(img_con_bandas, rango["color"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Ordenar bandas detectadas por posición horizontal
    bandas_detectadas = sorted(bandas_detectadas, key=lambda b: b[0])

    # Graficos demostrativos del proceso
    if mostrar_graficos_2 == True:
        plt.figure(figsize=(15, 7))
        plt.imshow(img_con_bandas)
        plt.title('Bandas de colores detectadas')
        plt.axis('off')
        plt.show()   

    return img_con_bandas, bandas_detectadas


# Verifica si la resistencia está orientada correctamente (banda de tolerancia a la derecha).
# Si no, gira la imagen 180° y ajusta las posiciones de las bandas.
# Devuelve: 
#  - Imagen ajustada si estaba invertida, o la original si ya estaba correcta.
#  - Lista de las bandas ajustadas en orden correcto.
def verificar_sentido_resistencia(img, bandas_detectadas):

    if len(bandas_detectadas) < 2:
        # Si no hay suficientes bandas para determinar la orientación
        return img, bandas_detectadas
    # Extraer las posiciones x de las bandas
    posiciones_x = [banda[0] for banda in bandas_detectadas]

    # Verificar si la resistencia está invertida (la última banda debe estar a la derecha)
    if min(posiciones_x) > 130:  
        # Girar la imagen 180°
        img_rotada = cv2.rotate(img, cv2.ROTATE_180)

        # Ajustar las posiciones de las bandas
        ancho_imagen = img.shape[1]
        bandas_ajustadas = [(ancho_imagen - x, color) for x, color in bandas_detectadas]

        # Ordenar las bandas ajustadas de izquierda a derecha
        bandas_ajustadas = sorted(bandas_ajustadas, key=lambda b: b[0])
        
        # Graficos demostrativos del proceso
        if mostrar_graficos_2 == True:
            titles_aux = ['Original' , 'Rotada']
            images_aux = [img, img_rotada]
            plt.figure(figsize=(14, 7))
            for i in range(2):
                plt.subplot(1, 2, i+1)
                cmap = 'gray' if i < 4 else None
                plt.imshow(images_aux[i], cmap=cmap)
                plt.title(titles_aux[i])
                plt.axis('off')
            plt.tight_layout()
            plt.show()

        return img_rotada, bandas_ajustadas

    # Si la orientación es correcta, devolver los valores originales
    return img, bandas_detectadas

''' ===============================================================================================
    ==================== TERCERA ETAPA: Calculo de valores de las resistencias ==================== '''

# --- Código de colores comercial ---
codigo_colores = {
    "Negro": 0, "Marron": 1, "Rojo": 2, "Naranja": 3, 
    "Amarillo": 4, "Verde": 5, "Azul": 6, "Violeta": 7, 
    "Gris": 8, "Blanco": 9
}
multiplicadores = {
    "Negro": 1, "Marron": 10, "Rojo": 100, "Naranja": 1000,
    "Amarillo": 10000, "Verde": 100000, "Azul": 1000000,
    "Violeta": 10000000, "Gris": 100000000, "Blanco": 1000000000
}

# Calcula el valor de la resistencia en Ohms basado en los nombres de las bandas detectadas.
# Convierte automáticamente a kOhms o MOhms si el valor es grande.
#  - Recibe: Lista de tuplas con coordenadas x y nombres de colores detectados.
#  - Devuelve: Valor de la resistencia en formato legible con unidades.
def calcular_valor_resistencia_comercial(bandas_detectadas):

    if len(bandas_detectadas) < 3:
        return "Error: Se necesitan al menos tres bandas para calcular el valor."

    # Extraer nombres de colores detectados, ordenados por posición
    colores = [banda[1] for banda in bandas_detectadas]

    # Calcular el valor de la resistencia
    valor_resistencia = (codigo_colores[colores[0]] * 10 + codigo_colores[colores[1]]) * multiplicadores[colores[2]]

    # Convertir a kOhms o MOhms si el valor es grande
    if valor_resistencia >= 1e6:
        return f"{valor_resistencia / 1e6:.2f} MOhms"
    elif valor_resistencia >= 1e3:
        return f"{valor_resistencia / 1e3:.2f} kOhms"
    else:
        return f"{valor_resistencia} Ohms"

''' ===============================================================================================
    ================= CUARTA ETAPA: Funcion principal que inicia el procedimiento ================= '''

# --- Procesar múltiples resistencias ---
def procesar_resistencias(directorio):
    resultados = []
    patron = re.compile(r'^R\d+_a_out\.jpg$')  # Patrón para seleccionar imágenes del tipo Rx_a_out.jpg
    for archivo in os.listdir(directorio):
        if patron.match(archivo):  # Verifica si el nombre cumple con el patrón
            ruta = os.path.join(directorio, archivo)

            # Cargar la imagen
            imagen = cv2.imread(ruta)
            
            roi, _ = detectar_cuerpo_resistencia(imagen)
            if roi is not None:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                _, colores_detectados = detectar_colores_bandas(roi_hsv)

                # Verificar sentido de la resistencia
                roi_ajustada, colores_ajustados = verificar_sentido_resistencia(roi, colores_detectados)

                # Calcular el valor de la resistencia
                valor_resistencia = calcular_valor_resistencia_comercial(colores_ajustados)

                resultados.append({
                    "imagen": archivo,
                    "colores": colores_ajustados,
                    "valor": valor_resistencia
                })

    return resultados

''' =============================================================================================== '''



''' =============== PROCEDIMIENTO =============== '''

# --- Itera sobre todas las imágenes en la carpeta "Resistencias" ---
for ruta in glob.glob("Resistencias/*.jpg"):
    img = cv2.imread(ruta)  # Lee la imagen actual
    try:
        salida = perspectiva_vista_superior_hibrido(img)  # Procesa la imagen para obtener la vista superior
        nombre = ruta.split("\\")[-1].replace(".jpg", "_out.jpg")  # Genera un nombre para la imagen de salida
        cv2.imwrite(f"Resultados/{nombre}", salida)  # Guarda la imagen procesada en la carpeta de resultados
    except Exception as e:
        print(f"Error con {ruta}: {e}")  # Informa errores durante el procesamiento

# --- Proceso las imagenes para obtener los valores de las resistencias ---
directorio_imagenes = "Resultados"
resultados = procesar_resistencias(directorio_imagenes)

# --- Muestra resultados ---
for resultado in resultados:
    colores = ", ".join([color for _, color in resultado['colores']])  # Extraigo solo los nombres de los colores
    print('======================================================================')
    print(f"Imagen: {resultado['imagen']} \t Colores: {colores}")
    print(f"Valor de resistencia: {resultado['valor']}")
