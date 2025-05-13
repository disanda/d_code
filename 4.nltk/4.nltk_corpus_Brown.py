from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import string

files = brown.fileids()
print(files) #
#print(files[-10:])
print(f"Total documents: {len(files)}")
#doc_id = 'test/14826'


# 获取原始文本
text = brown.raw('ca01')
print(text[:200])  # 打印前 200 个字符
print(brown.categories(files))
#['acq', 'coffee', 'corn', 'crude', 'gold', 'grain', 'lumber', 'nat-gas', 
#'palm-oil', 'rice', 'rubber', 'ship', 'sugar', 'tin', 'trade', 'veg-oil', 'wheat']
print('categories----------')


# 获取单词列表
words = brown.words(files)
print(words[:10])  # 打印前 10 个单词, ['ASIAN', 'EXPORTERS', 'FEAR', 'DAMAGE', 'FROM', 'U', '.', 'S', '.-', 'JAPAN']
print(len(words))  # 统计总词数, 输出示例: 约 1,720,901 个单词
print('words----------')

# # 获取句子列表
sents = brown.sents(files)
print(sents[:3])  # 打印前 2 个句子
print(len(sents))  # 统计总句数, 输出示例: 约 54,716 个句子
print('sents----------')

# 词汇量（去重后的单词数）
vocab = set(words)
print(f"词汇量: {len(vocab)}")  # 输出示例: 约 13, 769 个唯一单词(词汇量)
print('----------')

# 平均句长（单词数）
avg_sent_len = sum(len(sent) for sent in sents) / len(sents)
print(f"平均句长: {avg_sent_len:.2f} 个单词")  # 输出示例: 约 33.57 个单词
print('----------')

# 词频统计
from nltk import FreqDist
freq_dist = FreqDist(words)
print(freq_dist.most_common(10))  
print('freq_dist----------')

# 分词
raw_text = text[:500]  # 取前 500 字符示例
tokens = word_tokenize(raw_text) # 格式区别sents()方法，其按列表元素分割句子, 句子内为粉刺
print(tokens[:10])  
print('tokens----------')

# 对前 100 个单词进行词性标注
from nltk import pos_tag
words = words[:100]
tagged_words = pos_tag(words)
print(tagged_words[:10])  # 打印前 10 个词的词性
print('tagged_words----------')

# 查找包含 "God" 的句子
god_sents = [sent for sent in sents if 'God' in sent]
print(f"包含 'God' 的句子数: {len(god_sents)}")  
print(god_sents[0])  # 打印第一个匹配的句子 
print('god_sents----------')

# 查找常见双词搭配
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
finder = BigramCollocationFinder.from_words(words)
finder.apply_freq_filter(10)  # 过滤出现次数少于 10 的搭配
collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
print(collocations)  
print('----------')

