import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    
    Formula:
    Expected = (Row Total * Column Total) / Grand Total
    Chi2 = sum((Observed - Expected)^2 / Expected)
    """
    # Convertimos la entrada a un arreglo de NumPy para operaciones vectorizadas
    C = np.array(C, dtype=float)
    
    # Paso 1: Calcular totales de filas, columnas y el total general
    # Usamos axis=1 para sumar horizontalmente y axis=0 para verticalmente
    row_totals = C.sum(axis=1)
    col_totals = C.sum(axis=0)
    grand_total = C.sum()
    
    # Paso 2: Calcular frecuencias esperadas
    # np.outer genera una matriz donde cada celda (i, j) es row_totals[i] * col_totals[j]
    expected = np.outer(row_totals, col_totals) / grand_total
    
    # Paso 3: Calcular el estadístico Chi-cuadrado
    # La operación es elemento a elemento (vectorizada)
    chi2 = np.sum((C - expected) ** 2 / expected)
    
    return float(chi2), expected

# --- Pruebas de validación ---
examples = [
    [[10, 20], [20, 10]],
    [[20, 30], [40, 60]],
    [[25, 25], [25, 25]]
]

for i, ex in enumerate(examples):
    c2, exp = chi2_independence(ex)
    print(f"Ejemplo {i+1}:")
    print(f"  chi2: {round(c2, 3)}")
    print(f"  expected:\n{exp}\n")