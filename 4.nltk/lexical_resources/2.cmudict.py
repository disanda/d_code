from nltk.corpus import cmudict
import pronouncing # 同类型库，获取发音

print('-----------')
entries = cmudict.entries()
print(entries[:10]) # 所有字母单词，及其对应音素
print(len(entries)) # 133737


print('-----------') # 收集所有音素
phonemes = set()
for word, pron in entries:
    phonemes.update(pron)
print(sorted(phonemes)) # 按字母顺序输出

print('-----------') # 查找某个词的音标
cmu_dict = cmudict.dict()

print("发音（全部）：", pronouncing.phones_for_word("hello")) # ['HH AH0 L OW1', 'HH EH0 L OW1'], 多音读法，有两个读音
print("发音（全部）：", cmu_dict["hello"]) # [['HH', 'AH0', 'L', 'OW1'], ['HH', 'EH0', 'L', 'OW1']

# 判断是否存在
print("是否存在 'world'：", "world" in cmu_dict)