import cv2
import mediapipe as mp
import time

# Initialiser la capture vidéo à partir de la caméra (0 pour la première caméra)
camera_capture = cv2.VideoCapture(0)

# Charger le module des mains de MediaPipe
media_pipe_hands = mp.solutions.hands
# Initialiser l'objet pour la détection des mains
hand_detector = media_pipe_hands.Hands()
# Initialiser l'objet pour dessiner les points et connexions des mains
drawing_utils = mp.solutions.drawing_utils

# Créer une spécification de dessin pour les connexions des mains (couleur verte)
connections_draw_spec = drawing_utils.DrawingSpec(color=(198, 189, 10), thickness=2, circle_radius=1)

# Initialiser les variables pour calculer les images par seconde (fps)
previous_time = 0
current_time = 0

# Boucle infinie pour traiter les images de la caméra en temps réel
while True:
    # Lire une image de la caméra
    success, image = camera_capture.read()
    # Convertir l'image en RGB (nécessaire pour MediaPipe)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Traiter l'image avec MediaPipe pour détecter les mains
    detection_results = hand_detector.process(image_rgb)

    # Si des mains ont été détectées
    if detection_results.multi_hand_landmarks:
        # Pour chaque main détectée
        for hand_landmarks in detection_results.multi_hand_landmarks:
            # Pour chaque point de repère de la main
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                # Obtenir les dimensions de l'image
                height, width, channels = image.shape
                # Calculer les coordonnées x et y du point de repère
                coord_x, coord_y = int(landmark.x * width), int(landmark.y * height)
                # Afficher les coordonnées du point de repère
                print(landmark_id, coord_x, coord_y)
                # Dessiner un cercle autour du point de repère
                cv2.circle(image, (coord_x, coord_y), 7, (217, 0, 234), cv2.FILLED)

            # Dessiner les points et les connexions des mains sur l'image
            drawing_utils.draw_landmarks(
                image, 
                hand_landmarks, 
                media_pipe_hands.HAND_CONNECTIONS, 
                connections_draw_spec, 
                connections_draw_spec
            )

    # Calculer les images par seconde (fps)
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    # Afficher les fps sur l'image
    cv2.putText(image, "FPS : "+str(int(fps)), (7, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (217, 0, 234), 3)

    # Afficher l'image traitée
    cv2.imshow("Image", image)
    # Attendre 1 ms pour l'affichage de l'image suivante
    cv2.waitKey(1)
