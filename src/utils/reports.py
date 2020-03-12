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


def generate_reports(res_a, res_b, res_c, model_name):
    configs = ["text only", "image only", "deep cca", "concatenated"]
    r = None
    for c in configs:
        task_a = res_a[c]["pred_cls_test"]
        task_b = res_b[c]["pred_cls_test"]
        task_c = res_c[c]["pred_cls_test"]
        r = generate_report(task_a, task_b, task_c, zipname="res_{}_{}.zip".format(model_name, "_".join(c.split())))
    return r

