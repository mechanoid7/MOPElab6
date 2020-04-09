import copy
import random
from math import sqrt

import numpy as np
import scipy.stats


def get_sum(*args):  # take int X or array Xi_values for multiplication
    args_enc = ()  # пустой кортеж для приёма расшифрованныйх значений
    encrypt = {  # таблица расшифровки
        4: (1, 2),
        5: (1, 3),
        6: (2, 3),
        7: (1, 2, 3),
        8: (1, 1),
        9: (2, 2),
        10: (3, 3)
    }
    for obj in args:  # розшифровуємо вхідні дані [4-10]
        if obj in encrypt:
            args_enc = args_enc + encrypt[obj]
        else:
            args_enc = args_enc + (obj,)

    summa = 0
    try:
        if args_enc[0] == "y":  # если функция первым аргументом принимает строку "у"(eng)
            if len(args_enc) == 1:  # только у
                summa = sum(my_list)
            else:  # ещё и Х
                for j in range(N):
                    sum_i_temp = 1
                    for i in range(len(args_enc) - 1):   # цикл для элементов, кроме первой "у"
                        sum_i_temp *= x_matrix[j][args_enc[i + 1] - 1]  # произведение всех возможных иксов
                    sum_i_temp *= my_list[j]  # умножение на "у"
                    summa += sum_i_temp

        elif len(args_enc) == 1:
            args = args_enc[0] - 1
            for obj in x_matrix:
                summa += obj[args_enc[0]-1]
        else:  # если функция принимает кортеж
            for obj in x_matrix:
                sum_i_temp = 1
                for i in range(len(args_enc)):
                    sum_i_temp *= obj[args_enc[i] - 1]   # перемножаем все Х из кортежа, для квадрата дважды добавляем в кортеж Х
                summa += sum_i_temp


    except Exception as e:
        print("def error >> ", e)
    return summa


def y_func(x1, x2, x3):  # уравнение функции отклика за условием, вариант №205
    return 10.0+6.7*x1+6.4*x2+8.0*x3+6.2*x1*x1+0.2*x2*x2+7.6*x3*x3+1.2*x1*x2+0.7*x1*x3+1.1*x2*x3+2.8*x1*x2*x3


def rand_var():
    return random.randint(0, 10)-5


x1min, x1max = -30, 20  # начальные условия
x2min, x2max = -25, 45
x3min, x3max = 25, 30
m = 3
N = 15

# num_of_coef = 11
num_of_coef_list = [4, 8, 11]

mx_max = (x1max + x2max + x3max) / 3
mx_min = (x1min + x2min + x3min) / 3
y_max = mx_max + 200
y_min = mx_min + 200

# y_list = np.random.randint(y_min, y_max, (N, m))  # create tab N*3 random 'y' in [y_min; y_max]

k = 3
l = sqrt(k)  # зоряне плече
x01 = (x1max+x1min)/2
x02 = (x2max+x2min)/2
x03 = (x3max+x3min)/2
dx1 = x1max-x01
dx2 = x2max-x02
dx3 = x3max-x03
ldx1_x01 = l*dx1+x01
_ldx1_x01 = (-1)*l*dx1+x01
ldx2_x02 = l*dx2+x02
_ldx2_x02 = (-1)*l*dx2+x02
ldx3_x03 = l*dx3+x03
_ldx3_x03 = (-1)*l*dx3+x03

x_matrix = [
    [x1min, x2min, x3min],
    [x1min, x2min, x3max],
    [x1min, x2max, x3min],
    [x1min, x2max, x3max],
    [x1max, x2min, x3min],
    [x1max, x2min, x3max],
    [x1max, x2max, x3min],
    [x1max, x2max, x3max],
    [_ldx1_x01, x02, x03],
    [ldx1_x01, x02, x03],
    [x01, _ldx2_x02, x03],
    [x01, ldx2_x02, x03],
    [x01, x02, _ldx3_x03],
    [x01, x02, ldx3_x03],
    [x01, x02, x03]
]

y_list = []  # генерация "уi"
for obj in x_matrix:
    y_elem_temp = y_func(obj[0], obj[1], obj[2])
    y_list.append([y_elem_temp+rand_var(), y_elem_temp+rand_var(), y_elem_temp+rand_var()])

print("Yi:", y_list)

for num_of_coef in num_of_coef_list:  # ймовірний перехід до рівняння регресії з урахуванням ефекту взаємодії та квадратичних членів
    while 1:  # цикл для возможного увеличения m+1
        def y_add_el():  # функция увеличения m
            for i in range(len(y_list)):
                y_list[i].append(y_func(x_matrix[i][0], x_matrix[i][1], x_matrix[i][2])+rand_var())

        my_list = []
        mx1 = 0
        mx2 = 0
        mx3 = 0

        for obj in y_list:  # создание списка my
            my_list.append(sum(obj)/len(obj))

        for obj in x_matrix:
            mx1 += obj[0]
            mx2 += obj[1]
            mx3 += obj[2]

        mx1 /= 8
        mx2 /= 8
        mx3 /= 8
        my = sum(my_list)/N


        """Generator coefs"""
        mi = []  # двумерный список всех коэффициентов
        mi_temp = []  # временный массив для хранения mi[1-15]
        # num_of_coef = 11
        for i in range(num_of_coef):
            mi_temp = []
            for j in range(num_of_coef):
                if i == j == 0:  # если оба элемента ==0, это первый элемент, который =N
                    mi_temp.append(N)
                elif i == 0:  # не передавать нули в функцию
                    mi_temp.append(get_sum(j))
                elif j == 0:  # не передавать нули в функцию
                    mi_temp.append(get_sum(i))
                else:
                    mi_temp.append(get_sum(i, j))
            mi.append(mi_temp)

        k = [get_sum("y"), get_sum("y", 1), get_sum("y", 2), get_sum("y", 3), get_sum("y", 4), get_sum("y", 5),
             get_sum("y", 6), get_sum("y", 7), get_sum("y", 8), get_sum("y", 9), get_sum("y", 10)]

        denominator = np.linalg.det(
            mi
        )

        mi_num_b0 = copy.deepcopy(mi)  # создание глубоких копий двумерного массива коэффициентов для подальшей
        mi_num_b1 = copy.deepcopy(mi)  # замены столбцов для нахождения коэффициентов
        mi_num_b2 = copy.deepcopy(mi)
        mi_num_b3 = copy.deepcopy(mi)
        mi_num_b12 = copy.deepcopy(mi)
        mi_num_b13 = copy.deepcopy(mi)
        mi_num_b23 = copy.deepcopy(mi)
        mi_num_b123 = copy.deepcopy(mi)
        mi_num_b11 = copy.deepcopy(mi)
        mi_num_b22 = copy.deepcopy(mi)
        mi_num_b33 = copy.deepcopy(mi)

        for i in range(len(mi_num_b0)):  # находим чеслитель для каждого коэффициента
            mi_num_b0[i][0] = k[i]
        numerator_b0 = np.linalg.det(
            mi_num_b0
        )

        for i in range(len(mi_num_b1)):
            mi_num_b1[i][1] = k[i]
        numerator_b1 = np.linalg.det(
            mi_num_b1
        )

        for i in range(len(mi_num_b2)):
            mi_num_b2[i][2] = k[i]
        numerator_b2 = np.linalg.det(
            mi_num_b2
        )

        for i in range(len(mi_num_b3)):
            mi_num_b3[i][3] = k[i]
        numerator_b3 = np.linalg.det(
            mi_num_b3
        )

        if num_of_coef > num_of_coef_list[0]:  # коэффициенты для уравнения с учётом эффекта взаимодействия
            for i in range(len(mi_num_b12)):
                mi_num_b12[i][4] = k[i]
            numerator_b12 = np.linalg.det(
                mi_num_b12
            )

            for i in range(len(mi_num_b13)):
                mi_num_b13[i][5] = k[i]
            numerator_b13 = np.linalg.det(
                mi_num_b13
            )

            for i in range(len(mi_num_b23)):
                mi_num_b23[i][6] = k[i]
            numerator_b23 = np.linalg.det(
                mi_num_b23
            )

            for i in range(len(mi_num_b123)):
                mi_num_b123[i][7] = k[i]
            numerator_b123 = np.linalg.det(
                mi_num_b123
            )

        if num_of_coef > num_of_coef_list[1]:  # коэффициенты для уравнения с учётом квадратических членов
            for i in range(len(mi_num_b11)):
                mi_num_b11[i][8] = k[i]
            numerator_b11 = np.linalg.det(
                mi_num_b11
            )

            for i in range(len(mi_num_b22)):
                mi_num_b22[i][9] = k[i]
            numerator_b22 = np.linalg.det(
                mi_num_b22
            )

            for i in range(len(mi_num_b33)):
                mi_num_b33[i][10] = k[i]
            numerator_b33 = np.linalg.det(
                mi_num_b33
            )

        b_coefs = [
            numerator_b0/denominator,
            numerator_b1/denominator,
            numerator_b2/denominator,
            numerator_b3/denominator,
        ]
        if num_of_coef > num_of_coef_list[0]:  # так же варианты для уравнений регрессии
            b_coefs.extend([
                numerator_b12/denominator,
                numerator_b13/denominator,
                numerator_b23/denominator,
                numerator_b123/denominator])
        if num_of_coef > num_of_coef_list[1]:
            b_coefs.extend([
                numerator_b11 / denominator,
                numerator_b22 / denominator,
                numerator_b33 / denominator])

        b_coefs_str = ''
        for i in range(len(b_coefs)):  # вывод коэффициентов
            b_coefs_str += "b"+str(i)+": "+str("%.3f" % b_coefs[i])+"  "
        print(b_coefs_str)


        regression_equation_str = 'Рівняння регресії: y = '  # вывод уравнения
        if num_of_coef == num_of_coef_list[0]:
            regression_equation_list = [f"{b_coefs[0]:.2f}", f"{b_coefs[1]:+.2f}*x1", f"{b_coefs[2]:+.2f}*x2",
                                        f"{b_coefs[3]:+.2f}*x3"]
        elif num_of_coef == num_of_coef_list[1]:
            regression_equation_list = [f"{b_coefs[0]:.2f}", f"{b_coefs[1]:+.2f}*x1", f"{b_coefs[2]:+.2f}*x2",
                                        f"{b_coefs[3]:+.2f}*x3", f"{b_coefs[4]:+.2f}*x12", f"{b_coefs[5]:+.2f}*x13",
                                        f"{b_coefs[6]:+.2f}*x123"]
        else:
            regression_equation_list = [f"{b_coefs[0]:.2f}", f"{b_coefs[1]:+.2f}*x1", f"{b_coefs[2]:+.2f}*x2",
                                        f"{b_coefs[3]:+.2f}*x3", f"{b_coefs[4]:+.2f}*x12", f"{b_coefs[5]:+.2f}*x13",
                                        f"{b_coefs[6]:+.2f}*x123", f"{b_coefs[7]:+.2f}*x11", f"{b_coefs[8]:+.2f}*x22",
                                        f"{b_coefs[9]:+.2f}*x33"]
        for obj in regression_equation_list:
            regression_equation_str += obj
        print(regression_equation_str)


        # нахождение дисперсии
        S2 = []
        for i in range(len(y_list)):
            S2_temp = 0
            for j in range(len(y_list[0])):
                S2_temp += (y_list[i][j] - my_list[i]) ** 2
            S2.append(S2_temp)


        """KOHREN"""
        print("S2: ", S2)
        Gp = max(S2)/sum(S2)

        m = len(y_list[0])
        f1 = m-1
        f2 = N  # N=15
        q = 0.05

        Gt = [None, 0.4709, 0.3346, 0.276, 0.242, 0.216, 0.2034, 0.19, 0.18, 0.174, 0.167, 0.143, 0.114, 0.089, 0.067]
        # def gtest(f_obs, f_exp=None, ddof=0):
        #     f_obs = np.asarray(f_obs, 'f')
        #     k = f_obs.shape[0]
        #     f_exp = np.array([np.sum(f_obs, axis=0) / float(k)] * k, 'f') \
        #                 if f_exp is None \
        #                 else np.asarray(f_exp, 'f')
        #     g = 2 * np.add.reduce(f_obs * np.log(f_obs / f_exp))
        #     return g, scipy.stats.chisqprob(g, k - 1 - ddof)
        #
        # Gt = gtest(f1, f2, 0.95)

        print("Gp:", "%.4f" % Gp)
        print("Gt:", Gt[f1])

        if Gp < Gt[f1]:
            print("Дисперсія однорідна")
            break
        else:
            print("Дисперсія не однорідна, m+1")
            m += 1
            y_add_el()


    x_matrix_normal = [
        [1, -1, -1, -1],
        [1, -1, -1, 1],
        [1, -1, 1, -1],
        [1, -1, 1, 1],
        [1, 1, -1, -1],
        [1, 1, -1, 1],
        [1, 1, 1, -1],
        [1, 1, 1, 1],
        [1, -1*l, 0, 0],
        [1, l, 0, 0],
        [1, 0, -1*l, 0],
        [1, 0, l, 0],
        [1, 0, 0, -1*l],
        [1, 0, 0, l],
        [1, 0, 0, 0]
    ]

    """STUDENT"""
    def get_beta(i):
        summa = 0
        for j in range(N):
            summa += my_list[j]*x_matrix_normal[j][i]
        summa /= N
        return summa


    S2B = sum(S2)/N
    S2beta = S2B/(N*m)
    Sbeta = np.sqrt(S2beta)

    beta = []
    for i in range(len(x_matrix_normal[0])):
        beta.append(get_beta(i))

    t = []
    for i in range(len(x_matrix_normal[0])):
        t.append(abs(beta[i])/Sbeta)

    f3 = f1*f2

    t_tab = scipy.stats.t.ppf((1 + (1-q))/2, f3)
    print("t табличне:", t_tab)
    for i in range(len(x_matrix_normal[0])):
        if t[i] < t_tab:
            print(f"t{i}({t[i]:.3f})<t_tab, b{i}=0")
            b_coefs[i] = 0

    print(f"Таблиця коефіціентів:{b_coefs}")
    y_hat = []
    for i in range(N):  # подсчёт и вывод у с шляпокой с учётом вида уравнения регрессии
        if num_of_coef == num_of_coef_list[0]:
            y_hat.append(b_coefs[0] + b_coefs[1]*x_matrix[i][0] + b_coefs[2]*x_matrix[i][1] + b_coefs[3]*x_matrix[i][2])
        elif num_of_coef == num_of_coef_list[1]:
            y_hat.append(b_coefs[0] + b_coefs[1]*x_matrix[i][0] + b_coefs[2]*x_matrix[i][1] + b_coefs[3]*x_matrix[i][2] +
                         b_coefs[4]*x_matrix[i][0]*x_matrix[i][1] + b_coefs[5]*x_matrix[i][0]*x_matrix[i][2] +
                         b_coefs[6]*x_matrix[i][1]*x_matrix[i][2] + b_coefs[7]*x_matrix[i][0]*x_matrix[i][1]*x_matrix[i][2])
        else:
            y_hat.append(b_coefs[0] + b_coefs[1]*x_matrix[i][0] + b_coefs[2]*x_matrix[i][1] + b_coefs[3]*x_matrix[i][2] +
                         b_coefs[4]*x_matrix[i][0]*x_matrix[i][1] + b_coefs[5]*x_matrix[i][0]*x_matrix[i][2] +
                         b_coefs[6]*x_matrix[i][1]*x_matrix[i][2] + b_coefs[7]*x_matrix[i][0]*x_matrix[i][1]*x_matrix[i][2] +
                         b_coefs[8]*x_matrix[i][0]*x_matrix[i][0] + b_coefs[9]*x_matrix[i][1]*x_matrix[i][1] +
                         b_coefs[10]*x_matrix[i][2]*x_matrix[i][2])

        print(f"y{i+1}_hat = {y_hat[i]:.2f}")

    """FISHER"""

    d = 2
    f4 = N - d
    S2_ad = 0
    for i in range(num_of_coef):
        S2_ad += (m/(N-d)*((y_hat[i] - my_list[i])**2))

    Fp = S2_ad/S2B
    Ft = scipy.stats.f.ppf(1-q, f4, f3)
    print("Fp:", Fp)
    print("Ft:", Ft)
    if Fp > Ft:
        print("Рівняння регресії не адекватно оригіналу при рівні значимості 0,05")
        if num_of_coef == num_of_coef_list[0]:
            print("Переходимо до рівняння з урахуванням ефекту взаємодії.")
        elif num_of_coef == num_of_coef_list[1]:
            print("Переходимо до рівняння з урахуванням квадратичних членів.")
        # программа начинается с начала для другого уравнения регрессии
    else:
        print("Рівняння регресії адекватно оригіналу при рівні значимості 0,05")
        break  # конец программы, если уравнение адекватно оригиналу
