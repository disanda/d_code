from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
import string

storyname = 'bible-kjv.txt' # 'edgeworth-parents.txt'  'bible-kjv.txt'

# 获取原始文本
bible_text = gutenberg.raw(storyname)
print(bible_text[:100])  # 打印前 100 个字符, 输出示例: "[The King James Bible]\n\nThe Old Testament of ..."
print('----------')

# 获取单词列表
bible_words = gutenberg.words(storyname)
print(bible_words[:10])  # 打印前 10 个单词, 输出示例: ['[', 'The', 'King', 'James', 'Bible', ']', 'The', 'Old', 'Testament', 'of']
print(len(bible_words))  # 统计总词数, 输出示例: 约 1,010,654 个单词
print('----------')

# 获取句子列表
bible_sents = gutenberg.sents(storyname)
print(bible_sents[:2])  # 打印前 2 个句子: [['[', 'The', 'King', 'James', 'Bible', ']'], ['The', 'Old', 'Testament', 'of', 'the', 'Holy', 'Bible']]
print(len(bible_sents))  # 统计总句数, 输出示例: 约 30,103 个句子
print('----------')

# 词汇量（去重后的单词数）
vocab = set(bible_words)
print(f"词汇量: {len(vocab)}")  # 输出示例: 约 13, 769 个唯一单词(词汇量)
print('----------')

# 平均句长（单词数）
avg_sent_len = sum(len(sent) for sent in bible_sents) / len(bible_sents)
print(f"平均句长: {avg_sent_len:.2f} 个单词")  # 输出示例: 约 33.57 个单词
print('----------')

# 词频统计
from nltk import FreqDist
freq_dist = FreqDist(bible_words)
print(freq_dist.most_common(10))  
# 打印最常见的 10 个单词: [(',', 70509), ('the', 62103), (':', 43766), 
# ('and', 38847), ('of', 34480), ('.', 26160), ('to', 13396), ('And', 12846), ('that', 12576), ('in', 12331)]
print('----------')

# 分词
raw_text = bible_text[:500]  # 取前 500 字符示例
tokens = word_tokenize(raw_text) # 格式区别sents()方法，其按列表元素分割句子, 句子内为粉刺
print(tokens[:10])  # 输出示例: ['[', 'The', 'King', 'James', 'Bible', ']', 'The', 'Old', 'Testament', 'of']
print('----------')

# 移除标点和停用词
stop_words = set(stopwords.words('english') + list(string.punctuation))
cleaned_tokens = [w.lower() for w in tokens if w.lower() not in stop_words]
print(cleaned_tokens[:10])  # 输出示例: ['king', 'james', 'bible', 'old', 'testament', 'king', 'james', 'bible', 'first', 'book']
# with open('cleaned_bible_words.txt', 'w') as f:
#     f.write('\n'.join(cleaned_tokens)) # 保存清洗后的单词到文件
print('----------')

# 对前 100 个单词进行词性标注
from nltk import pos_tag
words = bible_words[:100]
tagged_words = pos_tag(words)
print(tagged_words[:10])  # 打印前 10 个词的词性
# [('[', 'VB'), ('The', 'DT'), ('King', 'NNP'), ('James', 'NNP'), 
#('Bible', 'NNP'), (']', 'VBZ'), ('The', 'DT'), ('Old', 'NNP'), ('Testament', 'NNP'), ('of', 'IN')]
print('----------')

# 查找包含 "God" 的句子
god_sents = [sent for sent in bible_sents if 'God' in sent]
print(f"包含 'God' 的句子数: {len(god_sents)}")  # 输出示例: 约 3,423 句
print(god_sents[0])  # 打印第一个匹配的句子 
# 输出示例: ['1', ':', '1', 'In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
print('----------')

# 查找常见双词搭配
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
finder = BigramCollocationFinder.from_words(bible_words)
finder.apply_freq_filter(10)  # 过滤出现次数少于 10 的搭配
collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
print(collocations)  
# [(',', 'and'), ('the', 'LORD'), ("'", 's'), ('of', 'the'), 
#('shall', 'be'), ('I', 'will'), ('in', 'the'), ('said', 'unto'), (';', 'and'), ('thou', 'shalt')]
print('----------')