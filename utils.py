import math
import csv
import os
import numpy as np

# Các hằng số CSDL & Thuật toán
database_file = 'Database.csv'
THRESHOLD = 0.20
LIVENESS_THRESHOLD = 0.045
EAR_THRESHOLD = 0.23

# Các điểm Mốc MediaPipe
NORMALIZE_PAIR = (234, 454)
FEATURE_PAIRS = [
    (10, 152), (33, 263), (133, 362), (61, 291), 
    (1, 4), (4, 152), (33, 61), (263, 291)
]

def calculate_3d_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)

def extract_3d_features(landmarks):
    norm_dist = calculate_3d_distance(landmarks[NORMALIZE_PAIR[0]], landmarks[NORMALIZE_PAIR[1]])
    if norm_dist == 0: return None
    vec = [calculate_3d_distance(landmarks[p1], landmarks[p2]) / norm_dist for p1, p2 in FEATURE_PAIRS]
    return np.array(vec)

def get_ear(landmarks):
    # Trái
    ear_l = calculate_3d_distance(landmarks[159], landmarks[145]) / calculate_3d_distance(landmarks[33], landmarks[133])
    # Phải
    ear_r = calculate_3d_distance(landmarks[386], landmarks[374]) / calculate_3d_distance(landmarks[362], landmarks[263])
    return (ear_l + ear_r) / 2.0

def check_liveness(landmarks):
    avg_cheek_z = (landmarks[234].z + landmarks[454].z) / 2.0
    return abs(avg_cheek_z - landmarks[1].z)

def load_database():
    names, vectors = [], []
    if os.path.exists(database_file):
        with open(database_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    names.append(row[0])
                    vectors.append(np.array([float(x) for x in row[1:]]))
    return names, vectors
