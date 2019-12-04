
Directory structure:
testpaper/
    bin/
    configs/            #配置文件
    packages/
        blob_kit/       #获取联通区域cc工具包
        rectify/        #试卷纠偏工具包
        svm_model_v0_2/ #svm字符分类器
        text_or_graph/  #文字/图像分类器
        bd_sdk.py       #百度识别工具
        imutils.py      #图像处理工具
        ti_div_v06.py   #综合应用
    tests/
        demo1.py #获取行首字符demo
        demo2.py #题目提取demo
    img/    测试试卷目录（部分）
    res/    测试结果目录（部分）
    README.txt


Requirements:
    opencv-python==3.4.5.20
    baidu-aip
    matplotlib
    tesserocr
    Rtree==0.8.3
    scikit-learn==0.21.2
    scikit-image==0.15.0

Notice
百度API使用的app账号密码可在packages/bd_sdk.py下修改，默认使用高精度
