import subprocess
import os
import shutil
import re
import logging
import json

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    return path

def dump2json(data, token, dir='./tmp'):
    if token is not str:
        token = str(token)

    f = open(os.path.join(dir, token+'.json'),'w')
    json.dump(data, f, indent=4, separators=(',', ': '))
    f.close()

def loadjson(token, dir='./tmp'):
    if token is not str:
        token = str(token)

    f = open(os.path.join(dir, token + '.json'), 'r')
    data = json.load(f)
    f.close()
    return data





def run_wait_finish(cmd, encoding='utf8', shell=False):
    """
    执行命令，等候命令结束，并输出返回值。
    默认 shell=False,要求cmd为字符串列表，包含可执行程序和它的参数。
    若cmd为带参数的字符串，则shell必须为True。好处是支持命令内 pipe。缺点是得不到实际命令的pid，因为获取的pid为sh的pid.
    :param cmd:
    :param encoding:
    :param shell: 当为True时，子进程为sh（bash）,sh 再启动程序。当cmd为带参数的字符串时，必须为True
    :return:
    """
    proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, encoding=encoding)
    data, _ = proc.communicate()
    return data


def run_no_wait(cmd, encoding='utf8', shell=False, timeout=1, kill=False):
    """
    执行命令，等待timeout时间，若命令结束，输出返回值，否则根据是否kill，返回目前结果，或空字符串。
    默认 shell=False,要求cmd为字符串列表，包含可执行程序和它的参数。
    若cmd为带参数的字符串，则shell必须为True。好处是支持命令内 pipe。缺点是得不到实际命令的pid，因为获取的pid为sh的pid.
    :param cmd:
    :param encoding:
    :param timeout:
    :param shell:
    :param kill: 注意，当设置kill时，shell 必须为 False。谨慎使用。
    :return:
    """
    proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, encoding=encoding)
    try:
        outs, errs = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if kill:
            print(proc.pid)
            proc.kill()
            outs, errs = proc.communicate()
        else:
            print(proc.pid)
            outs = ""
    return outs


def get_pid(key_words):
    """
    返回符合关键词的 pid 列表
    :param key_words: 字符串，或字符串列表。注意：当为列表时，按照其在列表中的顺序来匹配。
    :return:
    """
    cmd = ["ps", "-ef"]
    result = run_wait_finish(cmd, shell=False)

    ww = []
    if isinstance(key_words, str):
        ww.append(key_words)
    if isinstance(key_words, (list, tuple)):
        ww.extend(key_words)

    pattern_str = r'^[\w\d]+[\s]+([\d]+)[\s]+[\d]+.*'
    for w in ww:
        pattern_str = pattern_str + w + r'.*'
    pattern_str = pattern_str + r'$'

    pattern = re.compile(pattern_str, re.M)
    x = pattern.findall(result)
    return x

if __name__ == '__main__':
    pids = get_pid(['gst-launch-1.0','/dev/video0'])
    # x = get_pid(['sslocal','conf'])

    if pids:
        if(len(pids)>1):
            logging.warn('多个进程存在，请检查！')
        else:
            print(pids[0])
    else:
        # cmd = ["/usr/bin/gst-launch-1.0", "v4l2src device=/dev/video0 ! 'image/jpeg,width=1280,height=720'  ! rtpjpegpay ! udpsink clients='127.0.0.1:5004'"]
        cmd = ["/usr/bin/gst-launch-1.0", "v4l2src", "device=/dev/video0", "!", 'image/jpeg,width=1280,height=720',  "!", "rtpjpegpay", "!", "udpsink", "clients=127.0.0.1:5004"]
        run_no_wait(cmd)




