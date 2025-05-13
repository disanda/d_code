from nltk.corpus import treebank

file_ids = treebank.fileids() # 列出所有文件
print(file_ids[:5],'----------')  # 打印前 10 及后 10 个文件 ID

#['wsj_0001.mrg', 'wsj_0002.mrg', 'wsj_0003.mrg', 'wsj_0004.mrg', 'wsj_0005.mrg'] 
#--- ['wsj_0195.mrg', 'wsj_0196.mrg', 'wsj_0197.mrg', 'wsj_0198.mrg', 'wsj_0199.mrg']

#-------------------- 获取所有带 POS 标签的词 -----------------------
tagged_words = treebank.tagged_words()

#print(tagged_words[:10]) # 示例：打印前 10 个带 POS 标签的词

tagged_words_file = treebank.tagged_words(fileids='wsj_0002.mrg')
print(tagged_words_file[:10]) # 获取特定文件的带 POS 标签的词（例如 wsj_0001.mrg）


# ----------------------获取所有带 POS 标签的句子
tagged_sents = treebank.tagged_sents()

#print(tagged_sents[0]) # 示例：打印第一个句子的词和标签

tagged_sents_file = treebank.tagged_sents(fileids='wsj_0002.mrg')
print(tagged_sents_file[0]) # 获取特定文件的带 POS 标签的句子

# ------------------------------获取所有句法树
parsed_sents = treebank.parsed_sents()

# 示例：打印第一个句法树
print(parsed_sents[0])

# 或者以树形结构可视化（需要安装 graphviz 和 python-graphviz）
parsed_sents[0].draw()  # 弹出图形界面显示树

# 获取特定文件的句法树
parsed_sents_file = treebank.parsed_sents(fileids='wsj_0002.mrg')
print(parsed_sents_file[0])

parsed_sents[0].pretty_print()  # 终端打印树形结构, 或者保存为 PostScript 文件（需要 ghostscript）


# -----------------------------获取原始句子（不带标签）
sents = treebank.sents()

# 示例：打印第一个句子
print(sents[0])

# 获取特定文件的句子
sents_file = treebank.sents(fileids='wsj_0002.mrg')
print(sents_file[0])

# ---------------------------------统计所有 POS 标签的频率
from collections import Counter

tags = [tag for word, tag in treebank.tagged_words()]
tag_freq = Counter(tags)
print(tag_freq.most_common(10))  # 打印最常见的 10 个标签

# [('NN', 13166), ('IN', 9857), ('NNP', 9410), ('DT', 8165), ('-NONE-', 6592), 
# ('NNS', 6047), ('JJ', 5834), (',', 4886), ('.', 3874), ('CD', 3546)]

# ---------------------- conll2000 corpus
from nltk.corpus import conll2000

# 加载数据
train_data = conll2000.chunked_sents('train.txt')
test_data = conll2000.chunked_sents('test.txt')
print("First chunked sentence:\n", train_data[0])

# 提取原始句子（仅词）
original_sentences = [[word for word, pos in sent.leaves()] for sent in train_data]
for i, sent in enumerate(original_sentences[:5]): # 示例：打印前 5 个原始句子
    print(f"Sentence {i+1}: {sent}")


# 可视化分块树（需要图形界面）
train_data[0].draw()

# 统计训练集的 POS 标签
pos_tags = Counter(pos for word, pos, chunk in conll2000.iob_words('train.txt'))
print("Top 5 POS tags:", pos_tags.most_common(5))

# 统计训练集的分块标签
chunk_tags = Counter(chunk for word, pos, chunk in conll2000.iob_words('train.txt'))
print("Top 5 chunk tags:", chunk_tags.most_common(5))

# 获取训练集的词、POS 标签和分块标签
words_pos_chunks = conll2000.iob_words('train.txt')
print("First few tokens (word, POS, chunk):", words_pos_chunks[:10])

# 遍历分块树，提取 NP（名词短语）
for tree in conll2000.chunked_sents('train.txt')[:5]:  # 前 5 句
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':  # 检查名词短语
            print(subtree)

from nltk.chunk import RegexpParser

# 定义简单的分块规则（例如，NP 为连续的名词）
grammar = r"""
    NP: {<DT|JJ|NN.*>+}  # 名词短语：限定词、形容词、名词
"""

# 创建分块器
chunker = RegexpParser(grammar)

# 测试分块器
test_sent = test_data[0]  # 获取测试集的第一个句子
chunked = chunker.parse(test_sent)
print("Chunked result:\n", chunked)

# 可视化分块结果
chunked.draw()


#----------------------词频统计-
from nltk.corpus import brown, treebank, conll2000
from collections import Counter

# Brown POS 标签
brown_tags = Counter(tag for word, tag in brown.tagged_words())
print("Brown top 5 tags:", brown_tags.most_common(5))

# Treebank POS 标签
treebank_tags = Counter(tag for word, tag in treebank.tagged_words())
print("Treebank top 5 tags:", treebank_tags.most_common(5))

# CoNLL-2000 POS 标签
conll_tags = Counter(pos for word, pos, chunk in conll2000.iob_words())
print("CoNLL-2000 top 5 tags:", conll_tags.most_common(5))

