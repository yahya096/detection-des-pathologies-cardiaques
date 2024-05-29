#!/usr/bin/env python
# coding: utf-8

# In[29]:


get_ipython().system('pip install wfdb')


# In[30]:


import wfdb
import pandas as pd
record = wfdb.rdsamp('101')
df=pd.DataFrame(record[0],columns=record[1]['sig_name'])
df.to_csv('101.csv',index=False)


# In[31]:


data = pd.read_csv('101.csv')


# In[32]:


data


# In[33]:


from matplotlib.backends.backend_tkagg import (
     FigureCanvasTkAgg, NavigationToolbar2Tk)

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


MLII=data.iloc[1000:1400,0].values

root = tk.Tk()
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)  
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
toolbar.update()
toolbar.pack()
ax.plot(MLII, 'b')
canvas.draw()
root.mainloop()


frame = tk.Frame(root)
label = tk.Label(text = "Matplotlib + Tkinter!")
label.config(font=("Courier", 32))
label.pack()
frame.pack()



# In[34]:


max_values = np.partition(MLII, -3)[-3:]

# Find the two minimum values
min_values = np.partition(MLII, 2)[:2]

min_values


# In[35]:


import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Charger les données ECG
record = wfdb.rdrecord('101', physical=True)

# Récupérer le signal MLII
signal_MLII = record.p_signal[:, 0]

# Définir la taille de la fenêtre de lissage
window_size = 10  # Ajustez la taille de la fenêtre selon votre préférence

# Appliquer la moyenne mobile pour lisser le signal
smoothed_signal = np.convolve(signal_MLII, np.ones((window_size,))/window_size, mode='same')

# Afficher le signal original et le signal lissé
plt.figure(figsize=(12, 6))
plt.plot(signal_MLII[:1000], label='Signal MLII (Original)')
plt.plot(smoothed_signal[:1000], label='Signal MLII (Lissé)')
plt.xlabel('Temps (échantillons)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


# In[102]:


import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import wfdb

# Charger les données ECG depuis la base de données MIT-BIH
record = wfdb.rdrecord('101', physical=True)
signal_MLII = record.p_signal[:400, 0]

# Fonction pour mettre à jour le graphique en fonction de la position du slider
def update_slider(val):
    position = int(val)
    ax.clear()
    ax.plot(signal_MLII)
    ax.axvline(x=position, color='red', linestyle='--')
    # Afficher les coordonnées du point
    coord_label.config(text=f"Coordonnées : ({position}, {signal_MLII[position]:.2f})")
    canvas.draw()

# Créer la fenêtre Tkinter
root = tk.Tk()
root.title("Slider sur un graphique ECG")

# Créer le graphique
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(signal_MLII)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Créer un cadre pour le slider
slider_frame = tk.Frame(root)
slider_frame.pack()
slider_label = tk.Label(slider_frame, text="Position du slider:")
slider_label.pack()

# Créer le slider
slider = tk.Scale(slider_frame, from_=0, to=len(signal_MLII)-1, orient="horizontal", command=update_slider)
slider.set(0)  # Position initiale du slider
slider.pack()

# Ajuster la position et l'apparence du slider dans le graphique
canvas.get_tk_widget().place(x=100, y=300)  # Ajustez les coordonnées (x, y) selon votre graphique
slider_frame.place(x=100, y=350)  # Ajustez les coordonnées (x, y) selon votre graphique

# Étiquette pour afficher les coordonnées
coord_label = tk.Label(root, text="")
coord_label.pack()

# Lancer la fenêtre Tkinter
root.mainloop()


# In[36]:


pip install numpy matplotlib pandas keras scikit-learn


# In[37]:


pip install biosppy


# In[50]:


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

# Ajuster la position et l'apparence du slider dans le graphique
canvas.get_tk_widget().place(x=100, y=300)  # Ajustez les coordonnées (x, y) selon votre graphique
slider_frame.place(x=100, y=350)  # Ajustez les coordonnées (x, y) selon votre graphique

# Étiquette pour afficher les coordonnées
coord_label = tk.Label(root, text="")
coord_label.pack()

# Lancer la fenêtre Tkinter
root.mainloop()


# In[100]:


import numpy as np
from biosppy.signals import ecg
import wfdb

# Charger les données ECG depuis la base de données MIT-BIH
record = wfdb.rdrecord('101', physical=True)
MLII = record.p_signal[:2000, 0]  # Augmenter la longueur du signal ECG

# Fonction pour détecter les pics QRS
def detect_qrs_peaks(ecg_signal, sampling_rate):
    out = ecg.ecg(signal=ecg_signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = out['rpeaks']
    return qrs_indices

# Détection des pics QRS
qrs_indices = detect_qrs_peaks(MLII, record.fs)

# Fonction pour comparer les différences entre les pics QRS successifs
def compare_qrs_differences(qrs_indices, ecg_signal):
    differences = []
    for i in range(1, len(qrs_indices)):
        difference = ecg_signal[qrs_indices[i]] - ecg_signal[qrs_indices[i-1]]
        differences.append(difference)
    return differences

# Comparer les différences entre les pics QRS successifs
qrs_differences = compare_qrs_differences(qrs_indices, MLII)

# Afficher les différences entre les pics QRS successifs
print("Différences entre les pics QRS successifs:", qrs_differences)

# Condition pour déterminer si le patient est sain
is_healthy = all(abs(diff) < 0.2 for diff in qrs_differences)
if is_healthy:
    print("Le patient est sain.")
else:
    print("Le patient présente des anomalies cardiaques.")


# In[ ]:





# In[69]:


import numpy as np
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

# Fonction pour calculer la durée temporelle entre les pics QRS successifs
def calculate_qrs_durations(qrs_indices, sampling_rate):
    return np.diff(qrs_indices) / sampling_rate  # En secondes

# Détection des pics QRS
qrs_indices = detect_qrs_peaks(MLII, record.fs)

# Calculer les durées entre les pics QRS successifs
qrs_durations = calculate_qrs_durations(qrs_indices, record.fs)

# Calculer les différences successives entre les durées
differences = np.diff(qrs_durations)

# Afficher les durées et les différences
print("Durées entre les pics QRS successifs:", qrs_durations)
print("Différences successives entre les durées:", differences)

# Déterminer si le patient est sain ou malade
if np.all(np.abs(differences) <= 0.2):
    print("Le patient est sain.")
else:
    print("Le patient est malade.")


# In[71]:


import numpy as np
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

# Fonction pour calculer la durée temporelle entre les pics QRS successifs
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

# Afficher les durées et la fréquence cardiaque
print("Durées entre les pics QRS successifs:", qrs_durations)
print("Fréquence cardiaque:", heart_rate, "bpm")
# Déterminer si le patient est sain ou malade en fonction de la fréquence cardiaque
if 60 <= heart_rate <= 100:
    print("Le patient est sain.")
else:
    print("Le patient est malade.")


# In[72]:


import numpy as np
import matplotlib.pyplot as plt
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

# Afficher la forme d'onde des pics QRS
fig, ax = plt.subplots(figsize=(10, 6))

for qrs_index in qrs_indices:
    qrs_waveform = MLII[qrs_index-50:qrs_index+50]  # Récupérer la forme d'onde autour du pic QRS
    ax.plot(np.arange(qrs_index-50, qrs_index+50), qrs_waveform, color='blue')

ax.scatter(qrs_indices, MLII[qrs_indices], c='red', marker='o', label='QRS Peaks')
ax.set_title('Forme d\'onde des pics QRS')
ax.set_xlabel('Échantillons')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()


# In[75]:


import numpy as np
import matplotlib.pyplot as plt
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

# Fonction pour comparer les formes d'ondes des pics QRS
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

# Afficher les résultats
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(qrs_indices, correlations, marker='o', linestyle='-', color='blue')
ax.set_title('Comparaison des formes d\'ondes des pics QRS')
ax.set_xlabel('Index du pic QRS')
ax.set_ylabel('Corrélation avec la première forme d\'onde')
plt.show()

# Seuil de corrélation pour la décision
seuil_correlation = 0.9  # À ajuster en fonction de vos besoins

# Décision sur la santé de l'individu
if np.mean(correlations) >= seuil_correlation:
    print("Le patient est sain.")
else:
    print("Le patient est malade.")


# In[78]:


import numpy as np
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

# Détection de la fibrillation auriculaire (FA)
def detect_afib(qrs_indices, ecg_signal, sampling_rate):
    rr_intervals = np.diff(qrs_indices) / sampling_rate
    heart_rate = 60 / np.mean(rr_intervals)  # En battements par minute
    
    # Seuil de fréquence cardiaque pour la fibrillation auriculaire (FA) (exemple : > 300 bpm)
    seuil_afib = 300
    
    if np.max(rr_intervals) < 1/seuil_afib:  # Si le plus court intervalle est inférieur au seuil
        return True
    else:
        return False

# Détection de la tachycardie
def detect_tachycardia(qrs_indices, sampling_rate):
    rr_intervals = np.diff(qrs_indices) / sampling_rate
    heart_rate = 60 / np.mean(rr_intervals)  # En battements par minute
    
    # Seuil de fréquence cardiaque pour la tachycardie (exemple : > 100 bpm)
    seuil_tachycardia = 100
    
    if heart_rate > seuil_tachycardia:
        return True
    else:
        return False

# Appliquer les algorithmes de détection d'arythmies
is_afib = detect_afib(qrs_indices, MLII, record.fs)
is_tachycardia = detect_tachycardia(qrs_indices, record.fs)

# Afficher les résultats
print("Détection de la fibrillation auriculaire:", is_afib)
print("Détection de la tachycardie:", is_tachycardia)


# In[ ]:





# In[91]:


import numpy as np
from scipy.signal import find_peaks
from biosppy.signals import ecg
import wfdb

def detect_t_peaks(ecg_signal, qrs_indices, sampling_rate):
    # Utiliser la moyenne des pics R comme hauteur de seuil pour les pics T
    threshold = np.mean(ecg_signal[qrs_indices])
    
    # Détecter les pics T
    t_peaks_indices, _ = find_peaks(ecg_signal, height=threshold, distance=sampling_rate * 0.6)
    
    return t_peaks_indices

# Charger les données ECG depuis la base de données MIT-BIH
record = wfdb.rdrecord('101', physical=True)
MLII = record.p_signal[:2000, 0]  # Augmenter la longueur du signal ECG

# Détection des pics QRS
out = ecg.ecg(signal=MLII, sampling_rate=record.fs, show=False)
qrs_indices = out['rpeaks']

# Détection des pics T
t_peaks_indices = detect_t_peaks(MLII, qrs_indices, record.fs)

# Afficher les résultats
print("Indices des pics T:", t_peaks_indices)

   


# In[94]:


# Calculer les intervalles entre les pics T
t_peaks_intervals = np.diff(t_peaks_indices)

# Calculer la variabilité des intervalles entre les pics T (TIR)
tir_variation = np.std(t_peaks_intervals)

# Afficher les résultats
print("Variabilité des intervalles entre les pics T (TIR):", tir_variation)

# Décision sur la santé cardiaque
seuil_tir_variation = 10  # À ajuster en fonction de vos besoins
if tir_variation < seuil_tir_variation:
    print("L'individu est sain.")
else:
    print("L'individu est malade.")


# In[95]:


# Calculer la durée des pics T
t_peaks_durations = np.diff(t_peaks_indices) / record.fs

# Afficher les durées des pics T
print("Durées des pics T (en secondes):", t_peaks_durations)

# Comparaison des durées des pics T
seuil_variation_duree = 0.05  # À ajuster en fonction de vos besoins

# Vérifier si la variation de durée dépasse le seuil
if np.max(np.abs(np.diff(t_peaks_durations))) > seuil_variation_duree:
    print("L'individu présente une variation significative des durées des pics T. Consultez un professionnel de la santé.")
else:
    print("L'individu semble sain en termes de durées des pics T.")


# In[ ]:





# In[ ]:




