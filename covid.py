import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import copy

m = 122  # досліджуваний період
n = 245  # перший досліджуваний день

N = [10669709, 60297396, 126264931, 51709098]  # населення Чехії, Італії, Японії та Китаю 2019
N_0 = [10629928, 60421760, 126529100, 51606633]  # населення 2018
M_0 = [113099, 639153, 1351035, 320597]  # deaths in 2019
G = [109899, 446201, 921734, 372306]  # birth rate 2019

sliced_growth = [[], [], []]  # confirmed death recovered but by day growth

begin = [[], [], []]

confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
dead = pd.read_csv('time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('time_series_covid19_recovered_global.csv')


def data_process(dataset):
    dataset.rename(columns={'Country/Region': 'Country'}, inplace=True)
    dataset['Country'].replace({"Korea, South": "South Korea"}, inplace=True)
    index_Korea = dataset.loc[dataset['Country'] == 'South Korea'].index[0]
    index_Czechia = dataset.loc[dataset['Country'] == 'Czechia'].index[0]
    index_Japan = dataset.loc[dataset['Country'] == 'Japan'].index[0]
    index_Italy = dataset.loc[dataset['Country'] == 'Italy'].index[0]
    list_of_countries = [index_Czechia, index_Italy, index_Japan, index_Korea]
    return list_of_countries


list_of_country_index_confirmed_with_ds = [confirmed, data_process(confirmed)]  # датасет и нужные индексы для стран
list_of_country_index_dead_with_ds = [dead, data_process(dead)]
list_of_country_index_recovered_with_ds = [recovered, data_process(recovered)]


# для одной страны (почистить в конце) конф дед реков
# country 0 cz 1 it 2 jap 4 kor

def make_into_growth(list_of_country_index_ds, country, end_date):
    sliced = list_of_country_index_ds[0].loc[[list_of_country_index_ds[1][country]], '8/31/20':end_date]
    growth_by_day = np.diff(sliced.values, axis=1)
    begin_conditions = list_of_country_index_ds[0].loc[list_of_country_index_ds[1][country], '8/31/20']
    return growth_by_day, begin_conditions


def sir_one_country(country, end_date, m):
    growth_confirmed, begin_con_confirmed = make_into_growth(list_of_country_index_confirmed_with_ds, country, end_date)
    growth_dead, begin_con_dead = make_into_growth(list_of_country_index_dead_with_ds, country, end_date)
    growth_recovered, begin_con_recovered = make_into_growth(list_of_country_index_recovered_with_ds, country, end_date)
    I = [begin_con_confirmed - begin_con_dead - begin_con_recovered]
    R = [begin_con_recovered]
    M = [begin_con_dead]
    S = [(N[country] - (n - 1) * M_0[country] * N[country] / (N_0[country] * 365) + (n - 1) * G[country] * N[country] /
          (N_0[country] * 365) - I[0] - R[0] - M[0])]

    m_1 = m_3 = M_0[country] / (365 * N_0[country])
    gamma = (G[country] * N[country] / (365 * N_0[country]))
    for i in range(m - 1):
        S.append(int((1 - m_1) * S[i] - growth_confirmed[0][i] + gamma))
        M.append(int(M[i] + growth_dead[0][i]))
        R.append(int((1 - m_3) * R[i] + growth_recovered[0][i]))
        I.append(int(I[i] + growth_confirmed[0][i] - growth_recovered[0][i] - growth_dead[0][i]))

    return S, I, R, M, m_1, gamma


SIRM = sir_one_country(0, '12/31/20',
                       122)  # for 0 czechia, 1 - italy, 2 - japan, 3 - korea; end date of explored period, how many days

dates = []
start = datetime.datetime.strptime("01-09-2020", "%d-%m-%Y")
end = datetime.datetime.strptime("01-01-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    dates.append(date.strftime("%d-%m-%Y"))

croped_dates = [dates[i] if i % 10 == 0 else ' ' for i in range(len(dates))]
labels = ['Сприйнятливі до захворювання', 'Кількість інфікованих', 'Кількість одужалих', 'Померлі']


def draw(comp, i, var_label):
    plt.xlabel('Date')
    plt.ylabel('Number of people')
    plt.xticks(range(len(croped_dates)), croped_dates, rotation='vertical', size='small')
    plt.plot(dates, comp, label=var_label[i], color='black')
    plt.legend()
    plt.show()


# for i in range(4):
#   draw(SIRM[i], i, labels)


def calculate_mark_a(SIRM):
    mark_a = [np.array([[0],
                        [0],
                        [0]])]
    matrix_f = []
    matrix_g = []
    P = [np.zeros((3, 3))]
    F = []
    y = []
    for i in range(m):
        matrix_f.append(np.array([[-SIRM[0][i] * SIRM[1][i], 0, 0],
                                  [SIRM[0][i] * SIRM[1][i], -SIRM[1][i], -SIRM[1][i]],
                                  [0, SIRM[1][i], 0],
                                  [0, 0, SIRM[1][i]]]))

        matrix_g.append(np.array([[(1 - SIRM[4]) * SIRM[0][i] + SIRM[5]],
                                  [SIRM[1][i]],
                                  [(1 - SIRM[4]) * SIRM[2][i]],
                                  [SIRM[3][i]]]))
    for i in range(m - 1):
        F.append(P[i].dot(matrix_f[i].transpose()).dot(np.linalg.inv(matrix_f[i].dot(P[i]).dot(
            matrix_f[i].transpose()) + np.eye(4))))
        tmp = np.eye(3) - F[i].dot(matrix_f[i])
        tmp1 = tmp.dot(P[i]).dot(tmp.transpose())
        P.append(tmp1 + np.eye(3) + F[i].dot(F[i].transpose()))
    F.append(P[m - 1].dot(matrix_f[m - 1].transpose()).dot(np.linalg.inv(matrix_f[m - 1].dot(
        P[m - 1]).dot(matrix_f[m - 1].transpose()) + np.eye(4))))
    for i in range(m - 1):
        y.append(np.array([[SIRM[0][i + 1]],
                           [SIRM[1][i + 1]],
                           [SIRM[2][i + 1]],
                           [SIRM[3][i + 1]]]) - matrix_g[i])
    for i in range(m - 1):
        etw = mark_a[i] + F[i].dot(y[i] - matrix_f[i].dot(mark_a[i]))
        if etw[0] < 0:
            etw[0] = 0
        if etw[1] < 0:
            etw[1] = 0
        if etw[2] < 0:
            etw[2] = 0
        mark_a.append(etw)
    return mark_a


lab = ['Оптимальна оцінка альфа', 'Оптимальна оцінка бета', 'Оптимальна оцінка мю_2']

mark_a_for_certain_country = calculate_mark_a(SIRM)
list_mark_alpha_for_certain_country = []
list_mark_beta_for_certain_country = []
list_mark_mu2_for_certain_country = []
for i in range(m):
    list_mark_alpha_for_certain_country.append(mark_a_for_certain_country[i][0].tolist()[0])
    list_mark_beta_for_certain_country.append(mark_a_for_certain_country[i][1].tolist()[0])
    list_mark_mu2_for_certain_country.append(mark_a_for_certain_country[i][2].tolist()[0])
list_of_marks = [list_mark_alpha_for_certain_country, list_mark_beta_for_certain_country,
                 list_mark_mu2_for_certain_country]
#for i in range(3):
#    draw(list_of_marks[i], i, lab)


max_alpha = max(list_mark_alpha_for_certain_country)
max_beta = max(list_mark_beta_for_certain_country)
max_mu = max(list_mark_mu2_for_certain_country)

def forecast(SIRM):
    forecast_sirm = [[], [], [], []]
    S_forecast_plus = copy.deepcopy(SIRM[0])
    I_forecast_plus = copy.deepcopy(SIRM[1])
    R_forecast_plus = copy.deepcopy(SIRM[2])
    M_forecast_plus = copy.deepcopy(SIRM[3])

    S_forecast_minus = copy.deepcopy(SIRM[0])
    I_forecast_minus = copy.deepcopy(SIRM[1])
    R_forecast_minus = copy.deepcopy(SIRM[2])
    M_forecast_minus = copy.deepcopy(SIRM[3])

    for i in range(121, 124):
        S_forecast_plus.append(int((1 - SIRM[4]) * S_forecast_plus[i] + SIRM[5]))
        I_forecast_plus.append(int(I_forecast_plus[i] + (max_alpha/100)* S_forecast_plus[i] * I_forecast_plus[i]))
        R_forecast_plus.append(int(R_forecast_plus[i] + max_beta/100 * I_forecast_plus[i] - SIRM[4] * R_forecast_plus[i]))
        M_forecast_plus.append(int(M_forecast_plus[i] + max_mu/100 * I_forecast_plus[i]))

        S_forecast_minus.append(int((1 - SIRM[4]) * S_forecast_minus[i] - (max_alpha/100) * S_forecast_minus[i] * I_forecast_minus[i] + SIRM[5]))
        I_forecast_minus.append(int(I_forecast_minus[i] - max_beta * I_forecast_minus[i] - max_mu * I_forecast_minus[i]))
        R_forecast_minus.append(int(R_forecast_minus[i] - SIRM[4] * R_forecast_minus[i]))
        M_forecast_minus.append(int(M_forecast_minus[i]))
    for k in range(122, 125):
        forecast_sirm[0].append(int(int(S_forecast_plus[k] + S_forecast_minus[k]) / 2))
        forecast_sirm[1].append(int(int(I_forecast_plus[k] + I_forecast_minus[k]) / 2))
        forecast_sirm[2].append(int(int(R_forecast_plus[k] + R_forecast_minus[k]) / 2))
        forecast_sirm[3].append(int(int(M_forecast_plus[k] + M_forecast_minus[k]) / 2))
    return forecast_sirm, S_forecast_plus, S_forecast_minus  # change here for diff conparment for country


f_sirm, r_f_p, r_f_m = forecast(SIRM)
dates_extended = []
start = datetime.datetime.strptime("01-09-2020", "%d-%m-%Y")
end = datetime.datetime.strptime("04-01-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    dates_extended.append(date.strftime("%d-%m-%Y"))

croped_dates_extended = [dates_extended[i] if i % 19 == 0 else ' ' for i in range(len(dates))]

croped_dates_extended[-1] = '03-01-2021'

SIRM_f = sir_one_country(0, '1/3/21', 125)


def draw_forecast(up, down, forecast, real):
    plt.xlabel('Date')
    plt.ylabel('Number of people')
    plt.xticks(range(len(croped_dates_extended)), croped_dates_extended, rotation='vertical', size='small')
    plt.plot(dates_extended, real, label='real data', linestyle='--', color='black')
    plt.plot(dates_extended[-5:], up[-5:], label='upper border', color='black')
    plt.plot(dates_extended[-5:], down[-5:], label='bottom border', color='black')
    plt.scatter(dates_extended[-3:], forecast[-3:], label='forecast', color='black', marker='+')
    plt.legend()
    # plt.savefig('f.png', dpi=400, bbox_inches='tight')
    plt.show()


draw_forecast(r_f_p, r_f_m, f_sirm[0], SIRM_f[0])

