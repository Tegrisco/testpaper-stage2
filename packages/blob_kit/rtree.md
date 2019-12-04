# 安装 Rtree
## windows
下载
[Rtree‑0.8.3‑cp36‑cp36m‑win_amd64.whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rtree)

pip install xxx.whl

安装

## Linux
1. 安装 libspatialindex

解压缩 libspatialindex-master.zip
进入目录
mkdir build
cd build
cmake ..
make
sudo make install 
2. 安装 Rtree
pip install Rtree

# 使用
使用函数
get_no_intersect_boxes(bw_image_path)