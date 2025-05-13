from nltk.corpus import words

word_list = words.words()
print(len(word_list),word_list[:10])
# 236736 ['A', 'a', 'aa', 'aal', 'aalii', 'aam', 'Aani', 'aardvark', 'aardwolf', 'Aaron']

print("applee" in word_list) # 测试单词是否在words中

print('-------------------------names')
from nltk.corpus import names
male_names = names.words('male.txt')
print(len(male_names), male_names[:10])
#2943 ['Aamir', 'Aaron', 'Abbey', 'Abbie', 'Abbot', 'Abbott', 'Abby', 'Abdel', 'Abdul', 'Abdulkarim']

names.words('female.txt') # 获取常见女性英文名
names.words() # 获取所有男女合并名
'jack' in names.words('male.txt') # 判断性别(是否是男性名字)


print('-------------------------stropwords')
from nltk.corpus import stopwords

# 获取英文停用词列表
stop_words = stopwords.words('english')
print("总数：", len(stop_words)) # 198
print("前10个停用词：", stop_words[:10])

stop_words_cn = stopwords.words('chinese')
print("总数：", len(stop_words_cn)) # 841
print("前30个停用词：", stop_words_cn[:30])

french_stopwords = stopwords.words('french') # 获取法语、葡萄牙语、英语和瑞典语的stopwords
print("总数：", len(french_stopwords)) # 157
print("前10个停用词：", french_stopwords[:10])

portuguese_stopwords = stopwords.words('portuguese')
print("总数：", len(portuguese_stopwords)) # 207
print("前10个停用词：", portuguese_stopwords[:10])

swedish_stopwords = stopwords.words('swedish')
print("总数：", len(swedish_stopwords))
print("前10个停用词：", swedish_stopwords[:10]) # 114
