import cv2
import torch
import numpy as np
import time
from datetime import datetime
import os
import pandas as pd
import pathlib

# Créer un dossier pour stocker les captures où des téléphones sont détectés
if not os.path.exists('phone_detections'):
    os.makedirs('phone_detections')

# Fonction pour sauvegarder les incidents dans un CSV
def save_incident(timestamp, confidence):
    df_path = 'phone_incidents.csv'
    
    # Créer un nouveau DataFrame si le fichier n'existe pas
    if not os.path.exists(df_path):
        df = pd.DataFrame(columns=['timestamp', 'confidence', 'image_path'])
    else:
        df = pd.read_csv(df_path)
    
    # Nom du fichier image
    image_filename = f"phone_detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
    image_path = os.path.join('phone_detections', image_filename)
    
    # Ajouter l'incident
    new_row = {'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'), 
               'confidence': confidence,
               'image_path': image_path}
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(df_path, index=False)
    
    return image_path


   
# Correction du problème PosixPath sur Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def load_model():
    # Charger le modèle custom avec des paramètres plus stricts
    from ultralytics import YOLO
    model = YOLO('best.pt')
    
    # Paramètres plus stricts pour réduire les faux positifs
    # Note: YOLO parameters are set differently in Ultralytics compared to YOLOv5
    return model

def start_detection(camera_id=0, alert_cooldown=10):
    model = load_model()
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la caméra")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_alert_time = 0
    
    # Classes que votre modèle peut détecter comme téléphones
    phone_classes = ['cell phone', 'phone', 'smartphone', 'mobile', 'cellphone']  # Essayez différentes orthographes
    
    # Paramètres ajustés pour mieux détecter les vrais téléphones
    min_confidence = 0.55  # Seuil un peu plus bas pour commencer
    min_size_ratio = 0.03  # Taille minimale réduite (3% de l'image)
    aspect_ratio_range = (0.3, 0.8)  # Plage de ratio élargie
    
    print("Détection de téléphone démarrée. Appuyez sur 'q' pour quitter.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLOv8 inference
        results = model(frame, conf=min_confidence)
        
        # Get the annotated frame
        annotated_frame = results[0].plot()
        current_time = time.time()
        
        # Check if any detections
        if len(results[0].boxes) > 0:
            # Convert results to numpy for easier processing
            detections = results[0].boxes.data.cpu().numpy()
            
            valid_phones = []
            for det in detections:
                xmin, ymin, xmax, ymax, conf, cls = det
                
                # Get class name from model
                class_name = results[0].names[int(cls)].lower()
                
                # Debug: afficher chaque détection
                print(f"Détection: {class_name} (conf: {conf:.2f}), taille: {xmax-xmin:.0f}x{ymax-ymin:.0f}")
                
                # Vérifier la classe (insensible à la casse)
                if not any(phone_class.lower() in class_name for phone_class in phone_classes):
                    continue
                    
                # Calculer taille et ratio
                w = xmax - xmin
                h = ymax - ymin
                aspect_ratio = w / h if h > 0 else 0
                
                # Vérifier taille minimale
                if w < width * min_size_ratio or h < height * min_size_ratio:
                    print(f"Taille trop petite: {w/width:.2f}x{h/height:.2f}")
                    continue
                    
                # Vérifier ratio aspect
                if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]):
                    print(f"Ratio inapproprié: {aspect_ratio:.2f}")
                    continue
                    
                valid_phones.append((xmin, ymin, xmax, ymax, conf, cls, class_name))
            
            if valid_phones:
                # Find the detection with the highest confidence
                best_det = max(valid_phones, key=lambda x: x[4])
                xmin, ymin, xmax, ymax, confidence, cls, class_name = best_det
                
                # Dessiner la détection (if not already drawn by plot())
                cv2.rectangle(annotated_frame, 
                            (int(xmin), int(ymin)),
                            (int(xmax), int(ymax)),
                            (0, 0, 255), 2)
                cv2.putText(annotated_frame, 
                          f"PHONE {confidence:.0%}",
                          (int(xmin), int(ymin)-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                          (0, 0, 255), 2)
                
                if current_time - last_alert_time > alert_cooldown:
                    timestamp = datetime.now()
                    image_path = save_incident(timestamp, confidence)
                    cv2.imwrite(image_path, frame)
                    print(f"Téléphone détecté! Confiance: {confidence:.2f}")
                    last_alert_time = current_time
        
        # Afficher le frame
        cv2.imshow('Détection Téléphone', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def generate_report():
    if not os.path.exists('phone_incidents.csv'):
        print("Aucun rapport disponible - pas d'incidents enregistrés.")
        return
    
    df = pd.read_csv('phone_incidents.csv')
    
    print("\n=== RAPPORT DE DÉTECTION DE TÉLÉPHONE ===")
    print(f"Nombre total d'incidents: {len(df)}")
    
    if not df.empty:
        print("\nIncidents par ordre chronologique:")
        for idx, row in df.iterrows():
            print(f"  {idx+1}. {row['timestamp']} - Confiance: {row['confidence']:.2f}")
    
    print("\nRapport généré le:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
def detect_phone_in_frame(model, frame, width=None, height=None):
    """
    Détecte un téléphone dans une image (frame).
    Retourne (True, confidence) si un téléphone est détecté, sinon (False, 0).
    """
    if width is None or height is None:
        height, width = frame.shape[:2]
    phone_classes = ['cell phone', 'phone', 'smartphone', 'mobile', 'cellphone']
    min_confidence = 0.55
    min_size_ratio = 0.03
    aspect_ratio_range = (0.3, 0.8)

    # Run YOLOv8 inference
    results = model(frame, conf=min_confidence)
    
    # Check if any detections
    if len(results[0].boxes) > 0:
        # Convert results to numpy for easier processing
        detections = results[0].boxes.data.cpu().numpy()
        
        for det in detections:
            xmin, ymin, xmax, ymax, conf, cls = det
            
            # Get class name from model
            class_name = results[0].names[int(cls)].lower()
            
            # Check if it's a phone class
            if not any(phone_class.lower() in class_name for phone_class in phone_classes):
                continue
            
            # Calculate width and height of detection
            w = xmax - xmin
            h = ymax - ymin
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Check size
            if w < width * min_size_ratio or h < height * min_size_ratio:
                continue
            
            # Check aspect ratio
            if not (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]):
                continue
            
            return True, float(conf)
    
    return False, 0.0
if __name__ == "__main__":
    print("=== Programme de détection de téléphone pendant un entretien ===")
    print("1. Démarrer la détection")
    print("2. Générer un rapport d'incidents")
    print("3. Quitter")
    
    choice = input("Choix: ")
    
    if choice == '1':
        camera_id = 0  # Utiliser 0 pour la webcam par défaut
        start_detection(camera_id=camera_id)
    elif choice == '2':
        generate_report()
    else:
        print("Programme terminé.")
