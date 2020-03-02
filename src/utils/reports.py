from zipfile import ZipFile
import numpy as np


def generate_report(task_a, task_b, task_c, save=True, zipname="res.zip"):
    report = ""
    task_a = task_a - 1
    for i in range(task_a.shape[0]):
        task_b_str = np.array2string(task_b[i], separator="")[1:-1]
        task_c_str = np.array2string(task_c[i], separator="")[1:-1]
        report += "{}_{}_{}\n".format(task_a[i], task_b_str, task_c_str)

    if save is True:
        print(report, file=open("answer.txt", 'w'))
        with ZipFile(zipname, 'w') as myzip:
            myzip.write("answer.txt")

    return report


