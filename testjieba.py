# encoding=utf-8

import jieba
import jieba.analyse
# jieba.enable_paddle()# 启动paddle模式。 0.40版之后开始支持，早期版本不支持
# strs=["我要先同各位小朋友講聲唔好意思","因為我想將呢本書獻畀一個大人","我噉做係有個好真切嘅理由","我喺一本書入面見過幅好得人驚嘅圖畫","然後我好孤獨噉生活落去"]
# for str in strs:
#     seg_list = jieba.cut(str,HMM=True,use_paddle=False) # 使用paddle模式
#     print("Paddle Mode: " + '/'.join(list(seg_list)))
# #jieba.add_word('會唔會')
# jieba.suggest_freq('會唔會', True)
# seg_list = jieba.lcut("我想知道佢會唔會睇得明", cut_all=False, HMM=True)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
# jieba.suggest_freq('然後', True)
jieba.load_userdict("./data/cantondict.txt")
seg_list = jieba.lcut_for_search("「唔該，幫我畫隻綿羊仔吖。」 「咩話？」")
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
# import jieba.posseg as pseg
# result = pseg.cut("然後我好孤獨噉生活落去")
#
# for w in result:
#     print(w.word, "/", w.flag, ", ", end=' ')
# print("\n" + "="*40)




