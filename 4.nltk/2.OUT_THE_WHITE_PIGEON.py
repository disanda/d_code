import nltk
from nltk.corpus import gutenberg

# 确保已下载所需的 NLTK 数据
nltk.download('gutenberg')

# 加载文本
text = gutenberg.raw('edgeworth-parents.txt')

# 定义故事的起始和结束标志
start_marker = 'THE WHITE PIGEON'
end_marker = 'THE BIRTHDAY PRESENT'

# 查找起始和结束位置
start_index = text.find(start_marker)
end_index = text.find(end_marker)

# 确保找到标志
if start_index != -1 and end_index != -1:
    # 提取故事内容
    story_text = text[start_index:end_index].strip() # 移除这些首尾字符 1.空格 ' '，2.制表符 \t，3.换行符 \n，4.回车符 \r

    # 保存到本地文件
    with open('the_white_pigeon.txt', 'w', encoding='utf-8') as f:
        f.write(story_text)

    print("《The White Pigeon》已成功保存到 'the_white_pigeon.txt' 文件中。")
else:
    print("未能找到指定的故事段落，请检查标志是否正确。")
