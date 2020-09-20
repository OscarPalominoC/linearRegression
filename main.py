import linearRegression as lr
import numpy as np

# Código main
def main():
    # Dataset: Estos valores los obtuve al tomar la lectura diaria del consumo de luz en mi hogar el mes de abril
    data_x = np.array([1,2,3,4,5, 6, 7 ,8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27])
    data_y = np.array([3356,3374,3388,3402,3416,3427,3442,3457,3468,3483,3498,3513,3527,3540,3555,3571,3585,3598,3611,3623,3636,3650,3664,3679,3693,3705,3717])
    
    # Obtenemos b1 y b2
    b = lr.estimate_b0_b1(data_x, data_y)
    print(f'Los valores son b0 = {b[0]}, b1 = {b[1]}')
    print(f'La fórmula de regresión lineal es: Y = {b[1]}X + {b[0]}')
    
    print('Graficando los valores')
    lr.plot_regression(data_x, data_y, b)


if __name__ == '__main__':
    main()
    