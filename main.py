import numpy as np
import datetime
import csv
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import matplotlib.pyplot as plt


delay_dates = 5


def error_calculation(data, pred):
    avg_error = 0
    data_size = data.shape[0]
    for i in range(1, data_size):
        avg_error += (data[i] - pred[i]) ** 2
    avg_error /= (data_size - 1)
    return avg_error


def plot_curve(data1, data2, data3, label1, label2, label3, xlabel, ylabel,
               dataname):
    data_size = data1.shape[0]
    time = np.linspace(0, data_size, data_size)
    plt.plot(time, data1, label=label1, color='r')
    plt.plot(time, data2, label=label2, color='g')
    plt.plot(time, data3, label=label3, color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./{}.png'.format(dataname))
    # plt.show()
    plt.close()


def plot_e(data1, data2, label1, xlabel, ylabel, dataName):
    data_size = data1.shape[0]
    e = np.abs(data1 - data2)
    time = np.linspace(0, data_size, data_size)
    plt.plot(time, e, label=label1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./{}.png'.format(dataName))
    # plt.show()
    plt.close()


def plot_curve_single(data):
    data_size = data.shape[0]
    time = np.linspace(0, data_size, data_size)
    plt.plot(time, data, label='Bias')
    plt.xlabel('Time')
    plt.ylabel('Bias')
    plt.legend()
    # plt.show()
    plt.close()


def plot_curve_two(data1, data2, label1, label2, xlabel, ylabel, data_name):
    data_size = data1.shape[0]
    time = np.linspace(0, data_size, data_size)
    plt.plot(time, data1, label=label1)
    plt.plot(time, data2, label=label2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./{}.png'.format(data_name))
    # plt.show()
    plt.close()


def kalman_filter(data):
    data_size = data.shape[0]
    h = 1.0  # the time step
    my_filter = KalmanFilter(dim_x=3, dim_z=1)
    my_filter.x = np.array([[50.],
                            [0.01],
                            [0.01]])  # initial state
    my_filter.F = np.array([[1., h, 0.5*h**2],
                            [0., 1., h],
                            [0., 0., 1.]])  # state transition matrix
    my_filter.H = np.array([[1., 0., 0.]])  # Measurement function
    my_filter.P *= 1000.  # initial covariance matrix
    my_filter.R = 200000000  #0.48
    my_filter.Q = Q_discrete_white_noise(dim=3, dt=h, var=512)  # uncertainty
    print(my_filter.Q)
    estimation = np.zeros(data_size)
    # estimation is based on the current state
    for i in range(data_size):
        z = data[i]
        my_filter.predict()
        my_filter.update(z)
        estimation[i] = my_filter.x[0][0]
    # curve plot
    # plot_curve(data, estimation)
    return estimation


def date_str2num(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').toordinal()


def sum_by_date(data, col_num):
    if col_num == 2:
        data_frame = pd.DataFrame(data=data,
                                  columns=['date', 'twitter_numbers'])
        gp = data_frame.groupby(['date'])['twitter_numbers'].sum().reset_index()
        gp.rename(columns={'twitter_numbers': 'twitter_numbers_of_dates'},
                  inplace=True)
    elif col_num == 3:
        data_frame = pd.DataFrame(data=data,
                                  columns=['date', 'remain', 'leave'])
        gp = data_frame.groupby(['date'])['remain', 'leave'].mean().reset_index()
        gp.rename(columns={'remain': 'remain_sum', 'leave': 'leave_sum'},
                  inplace=True)
    gp_np = gp.to_numpy().astype(int)
    return gp_np


def data_extraction(address, usecols):
    if len(usecols) == 2:
        dateParse = lambda dates: pd.datetime.strptime(dates,
                                                       '%Y-%m-%d %H:%M:%S')
    elif len(usecols) == 3:
        dateParse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    rawdata = pd.read_csv(address,
                          parse_dates={'timeline': ['dates']},
                          date_parser=dateParse,
                          usecols=usecols)
    new_data = []
    for i in range(rawdata.shape[0]):
        # print('date:{}'.format(str(rawdata.iloc[i, 0])))
        date_convert = int(date_str2num(str(rawdata.iloc[i, 0])))
        # print('date_convert:{}'.format(date_convert))
        new_data.append(date_convert)
        if len(usecols) == 2:
            new_data.append(rawdata.iloc[i, 1].astype(np.int))
        elif len(usecols) == 3:
            new_data.append(rawdata.iloc[i, 1].astype(np.int))
            new_data.append(rawdata.iloc[i, 2].astype(np.int))
    new_data = np.array(new_data)
    new_data = new_data.reshape((-1, len(usecols)))
    sum_data = sum_by_date(new_data, len(usecols))
    return sum_data


def correlation_coefficient(data1, data2):
    coef = np.corrcoef(data1, data2)  # data1 : before/after, data2: obers
    return coef


def MSE(data1, data2):
    data_size = data1.shape[0]
    mse = sum((data1 - data2)**2)/data_size
    return mse


def E34(before, after):
    data3 = before - after
    return np.linalg.norm(data3)/np.linalg.norm(before)


if __name__ == '__main__':
    # read the csv files. The csv file provides the real data. The data will be
    # stored in a matrix.
    spt_leave_h = data_extraction('support_leave_hour.csv', [0, 2])
    spt_remain_h = data_extraction('support_remain_hour.csv', [0, 2])
    real_polling = data_extraction('real_polling.csv', [1, 5, 6])
    spt_leave = []
    spt_remain = []
    gt_polling = []
    for i in range(real_polling.shape[0]):
        pos_leave = -1
        pos_remain = -1
        for j in range(spt_leave_h.shape[0]):
            if real_polling[i][0] == spt_leave_h[j][0]:
                pos_leave = j
                break
        for j in range(spt_remain_h.shape[0]):
            if real_polling[i][0] == spt_remain_h[j][0]:
                pos_remain = j
                break
        if pos_leave != -1 and pos_remain != -1:
            gt_polling.append(real_polling[i][1])  # Remain
            gt_polling.append(real_polling[i][2])  # Leave
            spt_leave.append(spt_leave_h[pos_leave][1])
            spt_remain.append(spt_remain_h[pos_remain][1])
    spt_leave = np.array(spt_leave).reshape(-1, 1)
    spt_remain = np.array(spt_remain).reshape(-1, 1)
    gt_polling = np.array(gt_polling).reshape(-1, 2)
    # gt_polling = np.log(gt_polling)
    # before_kalman_remain = []
    # before_kalman_leave = []
    # for i in range(spt_remain.shape[0]):
    #     remain = int(spt_remain[i] / (spt_leave[i] + spt_remain[i]) * 100)
    #     before_kalman_remain.append(remain)
    #     leave = int(spt_leave[i] / (spt_leave[i] + spt_remain[i]) * 100)
    #     before_kalman_leave.append(leave)
    # before_kalman_remain = np.log(np.array(before_kalman_remain))
    # before_kalman_leave = np.log(np.array(before_kalman_leave))
    # before_kalman_remain = np.array(before_kalman_remain)
    # before_kalman_leave = np.array(before_kalman_leave)
    # after_kalman_remain = kalman_filter(before_kalman_remain)
    # after_kalman_leave = kalman_filter(before_kalman_leave)
    # plot_curve(before_kalman_remain, after_kalman_remain, gt_polling[:, 0],
    #            'Before Kalman, Remain', 'After Kalman, Remain', 'Observation',
    #            'Time', 'Log Percentage', 'log_remain')
    # print('Before Kalman:')
    # print(correlation_coefficient(gt_polling[:, 0], before_kalman_remain))
    # print('After Kalman:')
    # print(correlation_coefficient(gt_polling[:, 0], after_kalman_remain))
    kal_spt_leave = kalman_filter(spt_leave)
    kal_spt_remain = kalman_filter(spt_remain)
    kal_polling = []
    for i in range(spt_leave.shape[0]):
        remain = int(spt_remain[i] / (spt_leave[i] + spt_remain[i]) * 100)
        remain_kal = int(kal_spt_remain[i] /
                         (kal_spt_leave[i] + kal_spt_remain[i]) * 100)
        leave = int(spt_leave[i] / (spt_leave[i] + spt_remain[i]) * 100)
        leave_kal = int(kal_spt_leave[i] /
                        (kal_spt_leave[i] + kal_spt_remain[i]) * 100)
        kal_polling.append(remain)
        kal_polling.append(remain_kal)
        kal_polling.append(leave)
        kal_polling.append(leave_kal)
    kal_polling = np.array(kal_polling).reshape(-1, 4)
    # plot_curve(kal_polling[:, 0], kal_polling[:, 1], gt_polling[:, 0],
    #            'Before Kalman', 'After Kalman', 'Original', 'Time', 'Remain %',
    #            'Remain')
    plot_curve(kal_polling[:, 2], kal_polling[:, 3], gt_polling[:, 1],
               'Before Kalman', 'After Kalman', 'Original', 'Time', 'Leave %',
               'Leave')
    e1 = np.abs(gt_polling[:62-delay_dates, 0] - kal_polling[delay_dates:62, 0])
    e2 = np.abs(gt_polling[:62-delay_dates, 0] - kal_polling[delay_dates:62, 1])
    plot_curve_two(e1, e2, 'e1 (Before Kal.)', 'e2 (After Kal.)',
                   'time', 'e', 'e')
    print('Before Kalman, delay days:{}'.format(delay_dates))
    coef1 = correlation_coefficient(gt_polling[:62-delay_dates, 0],
                                    kal_polling[delay_dates:62, 0])
    print(coef1)
    print('After Kalman, delay days:{}'.format(delay_dates))
    coef2 = correlation_coefficient(gt_polling[:62-delay_dates, 0],
                                    kal_polling[delay_dates:62, 1])
    print(coef2)
    print('The performance correlation coefficient is improved by {}%'.format
          ((coef2[0][1]-coef1[0][1])*100/coef1[0][1]))
    mse_before_kal = MSE(gt_polling[:, 0], kal_polling[:, 0])
    mse_after_kal = MSE(gt_polling[:, 0], kal_polling[:, 1])
    print('The MSE improved from {} to {}'.format(mse_before_kal, mse_after_kal))
    e3 = E34(before=gt_polling[:62-delay_dates, 1], after=kal_polling[delay_dates:62, 2])
    e4 = E34(before=gt_polling[:62-delay_dates, 1], after=kal_polling[delay_dates:62, 3])
    print('e3 = {}, e4 = {}'.format(e3, e4))
