#!/usr/bin/env python
# coding: utf-8

!pip install wfdb
!pip install biosppy


#******************* Visualisation ***********************#
# In[1]:




import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from biosppy.signals import ecg
import wfdb

# Charger les données ECG depuis la base de données MIT-BIH
record = wfdb.rdrecord('101', physical=True)
MLII = record.p_signal[:2000, 0]  # Increase the length of the ECG signal

# Fonction pour détecter les pics QRS
def detect_qrs_peaks(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = out['rpeaks']
    return qrs_indices

# Détection des pics QRS
qrs_indices = detect_qrs_peaks(MLII, record.fs)

# Fonction pour mettre à jour le graphique en fonction de la position du slider
def update_slider(val):
    position = int(val)
    ax.clear()
    ax.plot(MLII)
    ax.scatter(qrs_indices, MLII[qrs_indices], c='red', marker='o', label='QRS Peaks')
    ax.axvline(x=position, color='green', linestyle='--', label='Slider Position')
    # Afficher les coordonnées du point
    coord_label.config(text=f"Coordonnées : ({position}, {MLII[position]:.2f})")
    ax.legend()
    canvas.draw()

# Créer la fenêtre Tkinter
root = tk.Tk()
root.title("Slider sur un graphique ECG avec QRS Peaks")

# Créer le graphique
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(MLII)
ax.scatter(qrs_indices, MLII[qrs_indices], c='red', marker='o', label='QRS Peaks')
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Créer un cadre pour le slider
slider_frame = tk.Frame(root)
slider_frame.pack()
slider_label = tk.Label(slider_frame, text="Position du slider:")
slider_label.pack()

# Créer le slider
slider = tk.Scale(slider_frame, from_=0, to=len(MLII)-1, orient="horizontal", command=update_slider)
slider.set(0)  # Position initiale du slider
slider.pack()


canvas.get_tk_widget().place(x=100, y=300)  
slider_frame.place(x=100, y=350) 

# Étiquette pour afficher les coordonnées
coord_label = tk.Label(root, text="")
coord_label.pack()

root.mainloop()

#********************************************************************************************************

#                                -- classer si une personne est malade ou non en se basant sur le rythme cardiaque --
# In[2]:
import numpy as np
from biosppy.signals import ecg
import wfdb


record = wfdb.rdrecord('101', physical=True)
MLII = record.p_signal[:2000, 0]  


def detect_qrs_peaks(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = out['rpeaks']
    return qrs_indices


def calculate_qrs_durations(qrs_indices, sampling_rate):
    return np.diff(qrs_indices) / sampling_rate  # En secondes

# Fonction pour calculer la fréquence cardiaque à partir des intervalles RR
def calculate_heart_rate(rr_intervals):
    heart_rate = 60 / np.mean(rr_intervals)  # En battements par minute
    return heart_rate

# Détection des pics QRS
qrs_indices = detect_qrs_peaks(MLII, record.fs)

# Calculer les durées entre les pics QRS successifs
qrs_durations = calculate_qrs_durations(qrs_indices, record.fs)

# Calculer la fréquence cardiaque à partir des intervalles RR
heart_rate = calculate_heart_rate(qrs_durations)


print("Durées entre les pics QRS successifs:", qrs_durations)
print("Fréquence cardiaque:", heart_rate, "bpm")

if 60 <= heart_rate <= 100:
    print("Le patient est sain.")
else:
    print("Le patient est malade.")

#                                        -- Utilisation des pics P, Q, R, S et T pour classifier si une personne est malade ou non en étudiant des cas de maladies --
# In[3]:
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from biosppy.signals import ecg
import wfdb


record = wfdb.rdrecord('101', physical=True)
MLII = record.p_signal[:2000, 0]  


def detect_qrs_peaks(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = out['rpeaks']
    return qrs_indices


def compare_qrs_waveforms(qrs_indices, ecg_signal, window_size=50):
    correlations = []
    reference_waveform = ecg_signal[qrs_indices[0]-window_size:qrs_indices[0]+window_size]
    for qrs_index in qrs_indices:
        qrs_waveform = ecg_signal[qrs_index-window_size:qrs_index+window_size]
        correlation = np.corrcoef(reference_waveform, qrs_waveform)[0, 1]
        correlations.append(correlation)
    return correlations

# Comparaison des formes d'ondes des pics QRS
correlations = compare_qrs_waveforms(qrs_indices, MLII)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(qrs_indices, correlations, marker='o', linestyle='-', color='blue')
ax.set_title('Comparaison des formes d\'ondes des pics QRS')
ax.set_xlabel('Index du pic QRS')
ax.set_ylabel('Corrélation avec la première forme d\'onde')
plt.show()


seuil_correlation = 0.9  


if np.mean(correlations) >= seuil_correlation:
    print("Le patient est sain.")
else:
    print("Le patient est malade.")


def detect_qrs_peaks(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = out['rpeaks']
    return qrs_indices


qrs_indices = detect_qrs_peaks(MLII, record.fs)

# Détection de la fibrillation auriculaire (FA)
def detect_afib(qrs_indices, ecg_signal, sampling_rate):
    rr_intervals = np.diff(qrs_indices) / sampling_rate
    heart_rate = 60 / np.mean(rr_intervals)  # En battements par minute
    
    # Seuil de fréquence cardiaque pour la fibrillation auriculaire (FA) 
    seuil_afib = 300
    
    if np.max(rr_intervals) < 1/seuil_afib:  
        return True
    else:
        return False

# Détection de la tachycardie
def detect_tachycardia(qrs_indices, sampling_rate):
    rr_intervals = np.diff(qrs_indices) / sampling_rate
    heart_rate = 60 / np.mean(rr_intervals)  # En battements par minute
    
    # Seuil de fréquence cardiaque pour la tachycardie 
    seuil_tachycardia = 100
    
    if heart_rate > seuil_tachycardia:
        return True
    else:
        return False


is_afib = detect_afib(qrs_indices, MLII, record.fs)
is_tachycardia = detect_tachycardia(qrs_indices, record.fs)


print("Détection de la fibrillation auriculaire:", is_afib)
print("Détection de la tachycardie:", is_tachycardia)

def detect_t_peaks(ecg_signal, qrs_indices, sampling_rate):
   
    threshold = np.mean(ecg_signal[qrs_indices])
    
    
    t_peaks_indices, _ = find_peaks(ecg_signal, height=threshold, distance=sampling_rate * 0.6)
    
    return t_peaks_indices



# Détection des pics T
t_peaks_indices = detect_t_peaks(MLII, qrs_indices, record.fs)


print("Indices des pics T:", t_peaks_indices)

# Calculer les intervalles entre les pics T
t_peaks_intervals = np.diff(t_peaks_indices)

# Calculer la variabilité des intervalles entre les pics T (TIR)
tir_variation = np.std(t_peaks_intervals)

# Afficher les résultats
print("Variabilité des intervalles entre les pics T (TIR):", tir_variation)

# Décision sur la santé cardiaque
seuil_tir_variation = 10  
if tir_variation < seuil_tir_variation:
    print("L'individu est sain.")
else:
    print("L'individu est malade.")


# Calculer la durée des pics T
t_peaks_durations = np.diff(t_peaks_indices) / record.fs

print("Durées des pics T (en secondes):", t_peaks_durations)


seuil_variation_duree = 0.05


if np.max(np.abs(np.diff(t_peaks_durations))) > seuil_variation_duree:
    print("L'individu présente une variation significative des durées des pics T. Consultez un professionnel de la santé.")
else:
    print("L'individu semble sain en termes de durées des pics T.")

