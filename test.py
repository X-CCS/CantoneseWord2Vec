# import os
# directory = os.path.dirname('data/test.txt')
#
# if not os.path.exists(directory):
#
#     os.makedirs(directory)
# with open('data/test.txt', 'w', encoding='utf8') as f:
#     f.write('测试' + '\n')

import jieba
#jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
jieba.load_userdict("./data/cantondict.txt")
test_sent = ["我要先同各位小朋友講聲唔好意思","因為我想將呢本書獻畀一個大人","我噉做係有個好真切嘅理由","我喺一本書入面見過幅好得人驚嘅圖畫","然後我好孤獨噉生活落去","我想知道佢會唔會睇得明"]
for i in range(len(test_sent)):
    words = jieba.cut(test_sent[i],HMM=True)
    print('/'.join(words) + '\n')
print("\n" + "="*40)

import pkuseg
seg = pkuseg.pkuseg(user_dict='./data/dict4pku.txt')
for i in range(len(test_sent)):
    words = seg.cut(test_sent[i])
    print('/'.join(words) + '\n')

print("\n" + "="*40)
import pkuseg
seg = pkuseg.pkuseg()
for i in range(len(test_sent)):
    words = seg.cut(test_sent[i])
    print('/'.join(words) + '\n')

print("\n" + "="*40)
