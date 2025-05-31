import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_and_save_statistics(data_path='heart.csv', output_dir='plots'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    
    # Mapear valores binarios para legibilidad
    binary_mappings = {
        'Sex': {1: 'Male', 0: 'Female'},
        'HeartDisease': {1: 'Yes', 0: 'No'},
        'FastingBS': {1: 'Yes', 0: 'No'},
        'ExerciseAngina': {1: 'Yes', 0: 'No'}
    }

    data_vis = data.copy()
    for col, mapping in binary_mappings.items():
        if col in data_vis.columns:
            data_vis[col] = data_vis[col].map(mapping)

    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    numeric_data = data.select_dtypes(include=np.number)
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # 2. Distributions for all columns
    for col in data_vis.columns:
        values = data_vis[col]
        plt.figure(figsize=(7, 5))

        if pd.api.types.is_numeric_dtype(values):
            sns.histplot(values, kde=True, bins=20, color='skyblue')
            plt.title(f"{col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Count")
        else:
            # Categorical: Pie Chart
            value_counts = values.value_counts()
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            plt.title(f"{col} Distribution")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col.lower()}_distribution.png")
        plt.close()

    print(f"Plots saved in: {output_dir}")

# Ejecutar
generate_and_save_statistics()






