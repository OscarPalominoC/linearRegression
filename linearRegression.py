import numpy as np
import matplotlib.pyplot as plt

# Función para hallar b0 y b1
def estimate_b0_b1(x, y):
    n = np.size(x)
    
    # Obtenemos los promedios de X y de Y
    m_x, m_y = np.mean(x), np.mean(y)
    
    # Calculando la sumatoria de XY y sumatoria de X*Xprom
    sumatoria_XY = np.sum((x-m_x)*(y-m_y))
    sumatoria_XXprom = np.sum(x*(x-m_x))
    
    # Coeficientes de regresión
    b_1 = sumatoria_XY/sumatoria_XXprom
    b_0 = m_y - b_1*m_x
    
    return (b_0, b_1)

# Función de graficado
def plot_regression(x, y, b):
    plt.scatter(x, y, color = 'g', marker = 'o', s=30)
    
    # Vector de prediciones
    y_pred = b[1]*x + b[0]
    plt.plot(x, y_pred, color='b')
    
    # Etiquetado
    plt.xlabel('X-Independiente')
    plt.ylabel('Y-Dependiente')
    
    plt.show()
