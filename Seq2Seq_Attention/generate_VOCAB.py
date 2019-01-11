# -*- coding:UTF-8 -*-

"""
@Author:Che_Hongshu
@Modify:2019.1.8
"""
import codecs
import collections
from operator import itemgetter

DATA_TYPE = "chinese"  # 将DATA_TYPE先后设置为chinese,english得到中英文ＶＯＣＡＢ文件

if DATA_TYPE == "chinese":  # 翻译语料的中文部分
    RAW_DATA = "./en-zh/train.txt.zh"
    VOCAB_OUTPUT = "zh.vocab"
    VOCAB_SIZE = 4000  #中文词汇表单词个数
elif DATA_TYPE == "english":  # 翻译语料的英文部分
    RAW_DATA = "./en-zh/train.txt.en"
    VOCAB_OUTPUT = "en.vocab"
    VOCAB_SIZE = 10000  #英文词汇表单词个数

counter = collections.Counter() #一个计数器，统计每个单词出现的次数

with codecs.open(RAW_DATA, "r", "utf-8") as f: #utf-8格式读取
    for line in f:
        for word in line.strip().split(): #line.strip().split()相当于把每一行的前后空格去掉，再根据空格分词生成list
            counter[word] += 1 #统计相同单词出现次数＋１
#  Counter 集成于 dict 类，因此也可以使用字典的方法，此类返回一个以元素为 key 、元素个数为 value 的 Counter 对象集合
#　依据key排序　itermgetter(1)为降序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)

#  转换成单词string的list
sorted_words_list = [x[0] for x in sorted_word_to_cnt]

#  加入句子结束符
sorted_words_list = ["<unk>", "<sos>", "<eos>"] + sorted_words_list

if len(sorted_words_list) > VOCAB_SIZE:
    sorted_words_list = sorted_words_list[:VOCAB_SIZE]

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words_list:
        file_output.write(word + '\n')
