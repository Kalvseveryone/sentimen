import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Data confusion matrix dari training sebelumnya
conf_matrix = np.array([[76, 17], [14, 93]])

# Set font and size
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4) 

# Create heatmap
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['Negatif (Prediksi)', 'Positif (Prediksi)'], 
                 yticklabels=['Negatif (Asli)', 'Positif (Asli)'],
                 annot_kws={"size": 16})

# Labels and Title
plt.title('Confusion Matrix - Sentimen Analisis\nAkurasi: 84.50%', fontsize=18, pad=20)
plt.xlabel('Nilai Prediksi', fontsize=16)
plt.ylabel('Nilai Aktual', fontsize=16)

# Save image
output_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Gambar berhasil disimpan di: {output_path}")
