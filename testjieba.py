#### This script is a testing script for Jieba Chinese Segmentation tool

import jieba
import jieba.analyse
jieba.enable_paddle()
jieba.load_userdict("./data/cantondict2.txt")
strs=["我要先同各位小朋友講聲唔好意思","因為我想將呢本書獻畀一個大人","我噉做係有個好真切嘅理由","我喺一本書入面見過幅好得人驚嘅圖畫","然後我好孤獨噉生活落去"]
for str in strs:
    seg_list = jieba.cut(str)
    print("Default Mode: " + '/'.join(list(seg_list)))
    seg_list = jieba.cut(str, HMM=True, use_paddle=False)
    print("HMM Mode:     " + '/'.join(list(seg_list)))
    seg_list = jieba.cut(str,HMM=False,use_paddle=True)
    print("Paddle Mode: " + '/'.join(list(seg_list)))
    seg_list = jieba.cut(str, HMM=True, use_paddle=True)
    print("Double Mode: " + '/'.join(list(seg_list)))
    print("\n" + "=" * 40)
jieba.add_word('會唔會')
jieba.suggest_freq('會唔會', True)
seg_list = jieba.lcut("我想知道佢會唔會睇得明", cut_all=False, HMM=True)
print("Default Mode: " + "/ ".join(seg_list))
jieba.suggest_freq('然後', True)
jieba.load_userdict("./data/cantondict.txt")
seg_list = jieba.cut_for_search("「唔該，幫我畫隻綿羊仔吖。」 「咩話？」")
print("Default Mode: " + "/ ".join(seg_list))





