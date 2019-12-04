import matplotlib.pyplot as plt

import cv2
from sklearn import datasets, svm, metrics
import os

import numpy as np
import pickle
import datetime as dt
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
#fetch original mnist dataset
from sklearn.datasets import fetch_mldata

import sys
sys.path.append('../packages/svm_model_v0_2/')
# import custom module
from mnist_helpers import *
import helpers as hp

sys.path.append('../packages/')
import imutils as imu
from bl_div_kit import adjust_img


class classifier:

    def __init__(self, model_path='svm-model.pickle'):
        # data_dir = '../dataset/svm'
        data_dir = '../packages/svm_model_v0_2'
        file = open(os.path.join(data_dir, model_path), 'rb')
        self.model = pickle.load(file)  # unpickle from binary file
        file.close()

    def predict_with_proba(self, gray, var_thres=0.01):
        im = imu.strip_white_boder(gray)
        im = adjust_img(im)
        sample = im.ravel() / 255.0

        var = np.var(self.model.predict_proba(sample.reshape(1, -1)))

        if var > var_thres:
            label = self.model.predict(sample.reshape(1, -1))
        else:
            label = [22]
        return label

    def predict(self, gray):
        im = imu.strip_white_boder(gray)
        im = adjust_img(im)
        sample = im.ravel() / 255.0

        return self.model.predict(sample.reshape(1, -1))


    def predict_proba(self, gray):
        im = imu.strip_white_boder(gray)
        im = adjust_img(im)
        sample = im.ravel() / 255.0

        return self.model.predict_proba(sample.reshape(1, -1))


def copy_augment():
    import shutil
    srcdir0 = '/home/kevin/PycharmProjects/autograde/dataset/svm/data-0810'
    dstdir0 = '/home/kevin/PycharmProjects/autograde/dataset/svm/data-0810-aug'

    for cls in range(22):
        srcdir = os.path.join(srcdir0, str(cls))
        dstdir = hp.mkdir(os.path.join(dstdir0, str(cls)))

        file_list = os.listdir(srcdir)

        aug_num = 5

        for file in file_list:
            if file.endswith('.jpg') or file.endswith('.png'):

                src_full = os.path.join(srcdir, file)
                dst_full = os.path.join(dstdir, file)
                print(src_full)

                shutil.copyfile(src_full, dst_full)

                gray = cv2.imread(src_full, 0)
                bw = imu.preprocess_bw(gray)
                dst_bw = dst_full[:-4]+'bw.jpg'
                cv2.imwrite(dst_bw, bw)

                for i in range(aug_num):
                    aug_full = dst_bw = dst_full[:-4]+'_'+str(i)+'.jpg'
                    if cls == 20: # for dot
                        aug = augment(gray, shift_low=0)
                    else:
                        aug = augment(gray, shift_low=-5)
                    # imu.imshow_(aug)
                    cv2.imwrite(aug_full, aug)

def resize_pad():

    srcdir0 = '/home/kevin/PycharmProjects/autograde/dataset/svm/data-0810'


    # srcdir0 = '/home/kevin/PycharmProjects/autograde/dataset/svm/data-0810-aug'
    dstdir0 = hp.mkdir('/home/kevin/PycharmProjects/autograde/dataset/svm/data-0810-28')

    # for cls in range(22):
    for cls in [22]:

        srcdir = os.path.join(srcdir0, str(cls))
        dstdir = hp.mkdir(os.path.join(dstdir0, str(cls)))

        file_list = os.listdir(srcdir)
        print(len(file_list))

        for file in file_list:
            if file.endswith('.jpg') or file.endswith('.png'):

                src_full = os.path.join(srcdir, file)
                dst_full = os.path.join(dstdir, file[:-4]+'.jpg')
                print(dst_full)

                gray = cv2.imread(src_full, 0)
                gray = imu.strip_white_boder(gray)
                result = adjust_img(gray)
                cv2.imwrite(dst_full, result)

def rotate_image(image, angle):
    # rotate the image to deskew it
    (h, w) = image.shape[:2]

    ww = int(w*1.2)
    hh = int(h*1.2)

    img = np.ones((hh,ww),dtype=np.uint8)*255
    img[(hh-h)//2:(hh-h)//2+h,(ww-w)//2:(ww-w)//2+w]=image

    center = (ww // 2, hh // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (ww, hh),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return rotated


def augment(gray, shift_low=-10):

    angle =(2*np.random.random()-1)*5.0
    # print(angle)
    rotated = rotate_image(gray, angle)

    # imu.imshow_(rotated)

    rotated = imu.strip_white_boder(rotated)
    # imu.imshow_(rotated)

    shift = np.random.randint(low=shift_low,high=50)
    gray = rotated
    if shift >= 0:
        gray = cv2.add(gray, shift)
    else:
        bw = imu.preprocess_bw_inv(gray)
        gray[bw>0] = gray[bw>0]+shift

    # print(shift)
    # if shift>0:
    #     gray = cv2.add(gray, shift)
    # else:
    #     gray = cv2.subtract(gray, shift)
    # imu.imshow_(gray)

    h, w = gray.shape
    hcut = min(2, h//5)
    wcut = min(2, w//5)

    top = np.random.randint(hcut+1)
    bottom = h-np.random.randint(hcut+1)
    left = np.random.randint(wcut+1)
    right = w-np.random.randint(wcut+1)

    return gray[top:bottom,left:right]



def all_strip_white():
    categories = range(22)  #不考虑其他类
    for c in categories:
        srcdir = os.path.join('../dataset/svm/data-0807', str(c))

        outdir = os.path.join('../dataset/svm/data-0810', str(c))


        if not os.path.exists(outdir):
            os.mkdir(outdir)

        papers = os.listdir(srcdir)
        # papers = ['120190703153331060.jpg']

        for paper in papers:
            if paper.endswith('.jpg') or paper.endswith('.png'):
                imgfile = os.path.join(srcdir, paper)
                outfile = os.path.join(outdir, paper)

                im = cv2.imread(imgfile, 0)
                imout = imu.strip_white_boder(im)

                cv2.imwrite(outfile, imout)




def pool_data_as_pickle():
    import random
    # categories = range(22)  #不考虑其他类
    categories = range(23)  #不考虑其他类

    num_each_max = 8000

    outdir = '../dataset/svm'
    data = []
    label = []
    for c in categories:
        srcdir = os.path.join('../dataset/svm/data-0810-28', str(c))



        if not os.path.exists(outdir):
            os.mkdir(outdir)

        papers_all = os.listdir(srcdir)
        # papers = ['120190703153331060.jpg']

        num_each = min(len(papers_all), num_each_max)
        papers = random.sample(papers_all, num_each)

        # papers = papers_all

        for paper in papers:
            if paper.endswith('.jpg') or paper.endswith('.png'):
                imgfile = os.path.join(srcdir, paper)
                print(imgfile)


                data.append(cv2.imread(imgfile, 0).ravel())
                label.append(c)

    data = np.array(data)
    label = np.array(label)

    file = open(os.path.join(outdir, 'svm-data.pickle'), 'wb')

    pickle.dump(data, file)  # pickle to binary file
    pickle.dump(label, file)  # pickle to binary file

    file.close()  # any file-like object will do



    return data, label


def pool_all_data_as_pickle():
    import random
    categories = range(23)  #不考虑其他类


    outdir = '../dataset/svm'
    data = []
    label = []
    for c in categories:
        srcdir = os.path.join('../dataset/svm/data-0810-28', str(c))



        if not os.path.exists(outdir):
            os.mkdir(outdir)

        papers = os.listdir(srcdir)
        # papers = ['120190703153331060.jpg']

        print(len(papers))

        for paper in papers:
            if paper.endswith('.jpg') or paper.endswith('.png'):
                imgfile = os.path.join(srcdir, paper)
                print(imgfile)


                data.append(cv2.imread(imgfile, 0).ravel())
                label.append(c)

    data = np.array(data)
    label = np.array(label)

    file = open(os.path.join(outdir, 'svm-data-all.pickle'), 'wb')

    pickle.dump(data, file)  # pickle to binary file
    pickle.dump(label, file)  # pickle to binary file

    file.close()  # any file-like object will do



def train():

    data_dir = '../dataset/svm'
    file = open(os.path.join(data_dir, 'svm-data-23-0813-milestone.pickle'), 'rb')

    data = pickle.load(file)  # unpickle from binary file
    label = pickle.load(file)  # unpickle from binary file

    file.close()

    show_some_digits(data, label)

    # data = data.astype(np.float32)/255

    X_data = data / 255.0
    Y = label

    # split data to train and test
    # from sklearn.cross_validation import train_test_split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

    ################ Classifier with good params ###########
    # Create a classifier: a support vector classifier

    # param_C = 5
    # param_gamma = 0.05
    # classifier = svm.SVC(C=param_C, gamma=param_gamma, probability=True, verbose=True)

    # classifier = svm.SVC(kernel='linear', probability=True, class_weight='balanced', verbose=True)
    classifier = svm.SVC(C = 0.7, kernel='linear', class_weight='balanced', verbose=True)


    # We learn the digits on train part
    start_time = dt.datetime.now()
    print('Start learning at {}'.format(str(start_time)))
    classifier.fit(X_train, y_train)
    end_time = dt.datetime.now()
    print('Stop learning {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print('Elapsed learning {}'.format(str(elapsed_time)))


    #######################################################

    file = open(os.path.join(data_dir, 'svm-model.pickle'), 'wb')

    pickle.dump(classifier, file)  # pickle to binary file

    file.close()  # any file-like object will do

    ########################################################
    # Now predict the value of the test

    file = open(os.path.join(data_dir, 'svm-model.pickle'), 'rb')

    classifier = pickle.load(file)  # unpickle from binary file
    file.close()

    expected = y_test
    predicted = classifier.predict(X_test)

    show_some_digits(X_test, predicted, title_text="Predicted {}")

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


def test_all():
    data_dir = '../dataset/svm'
    file = open(os.path.join(data_dir, 'svm-data.pickle'), 'rb')

    data = pickle.load(file)  # unpickle from binary file
    label = pickle.load(file)  # unpickle from binary file

    file.close()

    show_some_digits(data, label)

    # data = data.astype(np.float32)/255

    X_data = data / 255.0
    Y = label


    file = open(os.path.join(data_dir, 'svm-model.pickle'), 'rb')

    classifier = pickle.load(file)  # unpickle from binary file
    file.close()

    expected = Y
    predicted = classifier.predict(X_data)


    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))

    cm = metrics.confusion_matrix(expected, predicted)
    print("Confusion matrix:\n%s" % cm)

    plot_confusion_matrix(cm)

    print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


def calculate_prob_var():

    data_dir = '../dataset/svm'

    file = open(os.path.join(data_dir, 'svm-data.pickle'), 'rb')
    data = pickle.load(file)  # unpickle from binary file
    label = pickle.load(file)  # unpickle from binary file
    file.close()

    show_some_digits(data, label)

    X_data = data / 255.0
    Y = label

    file = open(os.path.join(data_dir, 'svm-model.pickle'), 'rb')
    classifier = pickle.load(file)  # unpickle from binary file
    file.close()

    expected = Y
    proba = classifier.predict_proba(X_data)

    vars = np.var(proba, axis=1)
    print(np.min(vars))
    print(np.max(vars))

    hp.dump2json(vars, 5)


def test_equal_delete():

    svm = classifier('svm-model-0811.pickle')

    c = 22
    srcdir = os.path.join('../dataset/svm/data-0810-28', str(c))

    papers = os.listdir(srcdir)
    # papers = ['120190703153331060.jpg']

    for paper in papers:
        if paper.endswith('.jpg') or paper.endswith('.png'):
            imgfile = os.path.join(srcdir, paper)
            print(imgfile)

            gray = cv2.imread(imgfile, 0)
            if svm.predict_with_proba(gray, var_thres=0.01) == 10:
                cv2.imshow("img", gray)
                k = cv2.waitKey(0)

                if k == ord('d'):
                    pass



if __name__ == "__main__":
    # resize_pad()
    # exit(0)
    # copy_augment()
    # exit(0)
    # pool_data_as_pickle()
    # exit(0)
    train()
    exit(0)
    # test_all()
    # exit(0)
    # pool_all_data_as_pickle()
    # exit(0)
    # calculate_prob_var()
    # exit(0)

    # test_equal_delete()
    # exit(0)