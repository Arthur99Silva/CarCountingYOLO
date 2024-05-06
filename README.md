# Car Counting using YOLO

Este projeto implementa um sistema de contagem de carros utilizando o algoritmo YOLO (You Only Look Once). O sistema é capaz de detectar carros em um vídeo e contabilizar sua passagem por uma linha de limite pré-definida.

Pré-requisitos:
  - Python 3.x
  - Bibliotecas Python:
     * Ultralytics YOLO
     * OpenCV (cv2)
     * CVZone
     * NumPy
     * SORT (Simple Online and Realtime Tracking)

   
# Instalação

1 - Clone este repositório:

git clone https://github.com/Arthur99Silva/CarCountingYOLO.git

2 - Instale as dependências:

pip install -r requirements.txt

3 - Baixe os pesos do modelo YOLO (yolov8n.pt) e o arquivo de máscara (mask2.png) e coloque-os nas pastas apropriadas conforme indicado no código.

# Uso

1 - Execute o script Python car_counting.py:

python Car-Counter.py

2 - O vídeo será carregado e os carros serão detectados e contados conforme cruzam a linha de limite especificada.

# Customização

Você pode ajustar os parâmetros do modelo YOLO, como confiança mínima (conf > 0.3) e classes detectadas, no código fonte.

Você também pode modificar a linha de limite e seus pontos de referência conforme necessário.
