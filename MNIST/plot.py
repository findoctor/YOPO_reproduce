import matplotlib.pyplot as plt
import pickle

def visualize(time_arr1, time_arr2,yopo_clean_err, yopo_robust_err, pgd_clean_err, pgd_robust_err):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #fig.suptitle(' Small CNN result on MNIST ', fontsize=20)
    yopo_clean_err = yopo_clean_err[::8]
    yopo_robust_err = yopo_robust_err[::8]
    time_arr2 = time_arr2[::8]

    # Retain part of PGD statics
    # print("Len of ele: " + str(len(time_arr1)) ) = 1876
    pgd_clean_err = pgd_clean_err[:340]
    pgd_robust_err = pgd_robust_err[:340]
    time_arr1 = time_arr1[:340]
    pgd_clean_err = pgd_clean_err[0::5]
    pgd_robust_err = pgd_robust_err[0::5]
    time_arr1 = time_arr1[0::5]

    ax1.plot(time_arr1, pgd_clean_err, color = 'red', label='PGD40 Clean Error')
    ax1.plot(time_arr1, pgd_robust_err, color = 'red', linestyle='-.', label='PGD40 Robust Error')

    ax1.plot(time_arr2, yopo_clean_err, color = 'g', label='YOPO-5-10 Clean Error')
    ax1.plot(time_arr2, yopo_robust_err, color = 'g', linestyle='-.', label='YOPO-5-10 Robust Error')

    plt.xlabel('MNIST Training Time(seconds)', fontsize=14)
    plt.ylabel('Error Rates(%)', fontsize=14)
    plt.legend(loc='upper right')
    plt.show()

file1 = open('yopo_clean_err.pkl', 'rb')
yopo_clean_err = pickle.load(file1)

file2 = open('yopo_robust_err.pkl', 'rb')
yopo_robust_err = pickle.load(file2)

file3 = open('yopo_time_arr.pkl', 'rb')
yopo_time_arr = pickle.load(file3)

file4 = open('pgd_clean_err.pkl', 'rb')
pgd_clean_err = pickle.load(file4)

file5 = open('pgd_robust_err.pkl', 'rb')
pgd_robust_err = pickle.load(file5)

file6 = open('pgd_time_arr.pkl', 'rb')
pgd_time_arr = pickle.load(file6)

visualize(pgd_time_arr, yopo_time_arr, yopo_clean_err, yopo_robust_err, pgd_clean_err, pgd_robust_err)

