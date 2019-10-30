import matplotlib.pyplot as plt
import pickle

def visualize(time_arr1, time_arr2,yopo_clean_err, yopo_robust_err, pgd_clean_err, pgd_robust_err):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    
    time_arr1 = [x/60 for x in time_arr1]
    time_arr2 = [x/60 for x in time_arr2]

    time_arr1 = time_arr1[::300]
    time_arr2 = time_arr2[::300]
    yopo_clean_err = yopo_clean_err[::300]
    yopo_robust_err = yopo_robust_err[::300]
    pgd_clean_err = pgd_clean_err[::300]
    pgd_robust_err = pgd_robust_err[::300]

    ax1.plot(time_arr1, pgd_clean_err, color = 'red', label='PGD10 Clean Error')
    ax1.plot(time_arr1, pgd_robust_err, color = 'red', linestyle='-.', label='PGD10 Robust Error')

    ax1.plot(time_arr2, yopo_clean_err, color = 'g', label='YOPO-5-3 Clean Error')
    ax1.plot(time_arr2, yopo_robust_err, color = 'g', linestyle='-.', label='YOPO-5-3 Robust Error')
    plt.legend(loc='upper right');

    plt.xlabel('CIFAR10 Training Time(minutes)', fontsize=14)
    plt.ylabel('Error Rates(%)', fontsize=14)
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

