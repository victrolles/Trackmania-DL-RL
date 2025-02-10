from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Charger le modèle YOLOv8 pré-entraîné (segmentation)
model = YOLO('yolo11n-seg.pt')  # Utiliser la version nano pour la rapidité

# Charger l'image
image_path = 'extras/tests/image.png'  # Chemin vers l'image d'entrée
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Effectuer la segmentation
results = model(image_rgb)

# Récupérer l'image segmentée
segmented_image = results[0].plot()  # Génère une image avec les masques segmentés

# Afficher le résultat
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Image segmentée avec YOLOv8')
plt.show()
