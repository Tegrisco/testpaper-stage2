import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import cv2
import joblib
import numpy as np
# from text_or_graph.get_featurecc import get_ccfeature
import os
# import src.blob_analysis as blob

import sys
sys.path.append('../packages/')
from text_or_graph.feature_box import get_ccfeature


def tell_me_text_or_graph(img, gray, boxes, classifier='./text_or_graph/bin/x.m', scalar='./text_or_graph/bin/scalar.m'):
    """
    由于效率原因，本函数一张图片只运行一次。
    后期如果需要分类的box很少，考虑不计算全图像中间结果。
    :param img:
    :param gray:
    :param boxes:
    :param classifier:
    :param scalar:
    :return:
    """

    clf = joblib.load(classifier)
    sca = joblib.load(scalar)

    features = get_ccfeature(img, gray, boxes)

    feature1 = sca.transform(features)

    return clf.predict(feature1)



def test(model, feature, boxes, img, imgpath):
    """Test model.

    Args:
        model: Model path.
        feature:特征.

    Returns:
        An integer result of class.

    """
    #
    #调用模型 clf = joblib.load(despath)
    # result = mlp.predict(X_test[0:3, 0:784])
    # print("img class:{}".format(result))
    # mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100, 100], activation='relu', alpha=1e-5, random_state=62)
    # joblib.dump(mlp, model)
    cc_label = []
    clf = joblib.load(model)
    # arr = []
    # arr1 = np.array(arr).reshape(1, -1)
    # result = mlp.predict(X_test[0:3, 0:784])
    index = 0
    for i in feature:
        arr = np.array(i).reshape(1, -1)
        label = clf.predict(arr)[0]
        if label == 1 or label == 2:
            cv2.rectangle(img, (boxes[index][0], boxes[index][1]), (boxes[index][0]+boxes[index][2], boxes[index][1]+boxes[index][3]), (0, 0, 255), 3)
        # if label == 2:
        #     cv2.rectangle(img, (boxes[index][0], boxes[index][1]), (boxes[index][0] + boxes[index][2], boxes[index][1] +boxes[index][3]), (255, 0, 0), 3)
        cc_label.append(label)
        index = index + 1
        # probs = clf.predict_proba(feature)[0]
        if label > 0:
            print('label为：{}'.format(label))
        cv2.imwrite(imgpath, img)
    return cc_label


def get_model(trainPath, testPath):
    dataset_train = sio.loadmat(trainPath)
    dataset_test = sio.loadmat(testPath)
    data_train = dataset_train['data']
    label_train = dataset_train['label']
    data_test = dataset_test['data']
    label_test = dataset_test['label']
    X_train = data_train
    y_train = label_train[0]
    X_test = data_test
    y_test = label_test[0]
    print('data_train:{}'.format(data_train.shape))
    print('label_train:{}'.format(label_train.shape))
    print('data_test:{}'.format(data_test.shape))
    print('label_test:{}'.format(label_test.shape))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2100, test_size=773, random_state=1)
    scaler = StandardScaler()
    change_function = scaler.fit(X_train)
    # max_abs_scaler = MaxAbsScaler()
    # X_train_maxabs = max_abs_scaler.fit_transform(X_train)
    # X_test_maxabs = max_abs_scaler.transform(X_test)
    # print(scaler.mean_)
    # print(scaler.scale_)
    X_train = change_function.transform(X_train)
    X_test = change_function.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4, solver='sgd', verbose=10, tol=1e-5, random_state=1)
    # mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[100, 100], activation='relu', alpha=1e-5, random_state=62, verbose=True)
    mlp.fit(X_train, y_train)
    desPath = r'./tmp/x.m'
    print('测试数据集得分：{:.2f}%'.format(mlp.score(X_test, y_test)*100))
    joblib.dump(mlp, desPath)
    mhj = r'../result/scalar.m'
    joblib.dump(change_function, mhj)
    print('模型已保存至：{}'.format(desPath))
    return desPath, mhj


def get_feature(mhj, featurePath):
    sca = joblib.load(mhj)
    features = get_ccfeature(img, gray, boxes)
    feature = sca.transform(features)
    joblib.dump(feature, featurePath)
    return featurePath


if __name__ == '__main__':

    from src.base import data_dir
    import src.helpers as hp
    import pickle
    from src.daxiaoti import box_whto2p


    books = ['biaozhundashijuan', 'jiangsumijuan', 'liangdiangeili']

    for book in books:

        srcdir = os.path.join(data_dir, 'waibu/' + book)
        baidu_dir = os.path.join('../dataset/baidu', book)
        CC_dir = os.path.join('../dataset/CC', book)

        out_base = hp.mkdir('../result/char_or_graph')

        outdir = hp.mkdir(os.path.join(out_base, book))

        papers = os.listdir(srcdir)
        # papers = ['120190703153513729.jpg']

        for paper in papers:
            if paper.endswith('.jpg'):
                print(os.path.join(srcdir, paper))
                img = cv2.imread(os.path.join(srcdir, paper))

                # boxes = hp.loadjson(paper, dir=CC_dir)
                # ocr = hp.loadjson(paper, dir=baidu_dir)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                bw2 = blob.preprocess_bw_inv(gray)
                cv2.imwrite('./tmp/xx.jpg', bw2)
                boxes = blob.get_no_intersect_boxes('./tmp/xx.jpg')

                label = tell_me_text_or_graph(img,gray,boxes)

                for i, box in enumerate(boxes):
                    if label[i] == 0:
                        loc2 = box_whto2p(box)
                        cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (255, 0, 0), 2)
                    else:
                        loc2 = box_whto2p(box)
                        cv2.rectangle(img, (loc2[0], loc2[1]), (loc2[2], loc2[3]), (0, 0, 255), 2)

                cv2.imwrite(os.path.join(outdir, paper), img)

                # break


