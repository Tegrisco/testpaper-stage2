from sys import platform

if platform == "linux" or platform == "linux2":
    # data_dir = '/mnt/lenovo/autograde/dataset'
    label_exe = r'../packages/blob_kit/bin/label_linux'
elif platform == "darwin":
    label_exe = r'../packages/blob_kit/bin/label_darwin'
elif platform == "win32":
    # data_dir = r'd:\autograde\dataset'
    label_exe = r'..\packages\blob_kit\bin\label.exe'






