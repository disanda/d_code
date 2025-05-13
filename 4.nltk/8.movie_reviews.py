# import nltk
# from nltk.corpus import movie_reviews
# import random

# print(len(movie_reviews.fileids()), movie_reviews.fileids()[:10]) # 2000, 1000 neg, 1000 pos

# print(movie_reviews.categories())

# print(movie_reviews.raw('neg/cv001_19502.txt')) # cv000_29416, cv001_19502 
#print(movie_reviews.raw('pos/cv999_13106.txt


# # 数据加载与简单示例
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]

# random.shuffle(documents)

# # 打印前一条影评与其标签
# print("Sample words:", documents[0][0][:20])
# print("Label:", documents[0][1])

# print('subjectivity---------------------')

# from nltk.corpus import subjectivity

# subj = subjectivity.sents(categories='subj')  # fileids, raw 获取主观句子
# obj = subjectivity.sents(categories='obj')   # 获取客观句子
# categories = subjectivity.categories()              # 返回 ['obj', 'subj']

# print(len(subj)," ".join(subj[0]))
# print(len(obj),obj[0])
# print(categories)


# print('product_reviews_1---------------------')

# from nltk.corpus import product_reviews_1

# for fileid in product_reviews_1.fileids():
#     print(fileid)
#     reviews = product_reviews_1.sents(fileid)
#     i = 0
#     for review in reviews:
#         text = " ".join(review)  # 将词列表转为字符串
#         i = i+1
#         print(i, text)

# for fileid in product_reviews_1.fileids():
#     print(fileid)
#     j = 0
#     features = product_reviews_1.features(fileid)
#     unique_features = sorted(set(feat for feat, score in features))
#     print(unique_features)
#     for feature in features:
#         feat, score = feature
#         j = j + 1
#         print(j, f"{feat}: {score}")

# import nltk
# from nltk.corpus import product_reviews_1

# nltk.download('product_reviews_1')

# # 取某个商品的评论，例如 Diaper_Champ
# reviews = product_reviews_1.parsed_reviews('Diaper_Champ.txt')

# # 查看第一个评论
# review = reviews[0]
# print("评论内容：", " ".join(review.text))
# print("情感标签：", review.sentiment())  # 输出 'pos' 或 'neg'

import nltk
from nltk.corpus import conll2000


# 加载数据
train_data = conll2000.chunked_sents('train.txt')
test_data = conll2000.chunked_sents('test.txt')

print(train_data[0])


