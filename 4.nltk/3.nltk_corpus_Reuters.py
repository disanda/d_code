from nltk.corpus import reuters, stopwords
from nltk.tokenize import word_tokenize
import string

reuters_files = reuters.fileids()
print(reuters_files[:5]) # ['test/14826', 'test/14828', 'test/14829',..., 'training/9994', 'training/9995']
print(reuters_files[-5:])
print(f"Total documents: {len(reuters_files)}")
doc_id = 'test/14826'


# 获取原始文本
reuters_text = reuters.raw(doc_id)
print(reuters_text[:200])  # 打印前 200 个字符
print(reuters.categories(reuters_files))
#['acq', 'coffee', 'corn', 'crude', 'gold', 'grain', 'lumber', 'nat-gas', 
#'palm-oil', 'rice', 'rubber', 'ship', 'sugar', 'tin', 'trade', 'veg-oil', 'wheat']
print('----------')


# 获取单词列表
reuters_words = reuters.words(reuters_files)
print(reuters_words[:10])  # 打印前 10 个单词, ['ASIAN', 'EXPORTERS', 'FEAR', 'DAMAGE', 'FROM', 'U', '.', 'S', '.-', 'JAPAN']
print(len(reuters_words))  # 统计总词数, 输出示例: 约 1,720,901 个单词
print('----------')

# # 获取句子列表
reuters_sents = reuters.sents(reuters_files)
print(reuters_sents[:2])  # 打印前 2 个句子
print(len(reuters_sents))  # 统计总句数, 输出示例: 约 54,716 个句子
print('----------')

# 词汇量（去重后的单词数）
vocab = set(reuters_words)
print(f"词汇量: {len(vocab)}")  # 输出示例: 约 13, 769 个唯一单词(词汇量)
print('----------')

# 平均句长（单词数）
avg_sent_len = sum(len(sent) for sent in reuters_sents) / len(reuters_sents)
print(f"平均句长: {avg_sent_len:.2f} 个单词")  # 输出示例: 约 33.57 个单词
print('----------')

# 词频统计
from nltk import FreqDist
freq_dist = FreqDist(reuters_words)
print(freq_dist.most_common(10))  
print('----------')

# 分词
raw_text = reuters_text[:500]  # 取前 500 字符示例
tokens = word_tokenize(raw_text) # 格式区别sents()方法，其按列表元素分割句子, 句子内为粉刺
print(tokens[:10])  
print('----------')

# 移除标点和停用词
stop_words = set(stopwords.words('english') + list(string.punctuation))
cleaned_tokens = [w.lower() for w in tokens if w.lower() not in stop_words]
print(cleaned_tokens[:10])  
print('----------')

# 对前 100 个单词进行词性标注
from nltk import pos_tag
words = reuters_words[:100]
tagged_words = pos_tag(words)
print(tagged_words[:10])  # 打印前 10 个词的词性
print('----------')

# 查找包含 "God" 的句子
god_sents = [sent for sent in reuters_sents if 'God' in sent]
print(f"包含 'God' 的句子数: {len(god_sents)}")  
print(god_sents[0])  # 打印第一个匹配的句子 
print('----------')

# 查找常见双词搭配
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
finder = BigramCollocationFinder.from_words(reuters_words)
finder.apply_freq_filter(10)  # 过滤出现次数少于 10 的搭配
collocations = finder.nbest(BigramAssocMeasures.likelihood_ratio, 10)
print(collocations)  
print('----------')