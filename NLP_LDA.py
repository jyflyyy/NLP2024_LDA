import csv
import os
import jieba
import random
import gensim
import numpy as np
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

def lda_data(textfile, flag):
        dream_text = open(textfile, 'r+', encoding='utf-8')
        text = list(filter(filter_para, dream_text.read().split("\n\n")))
        contents_new = map(lambda s: s.replace(' ', ''), text)
        contents_new = map(lambda s: s.replace('\u3000', ''), text)
        text = list(contents_new)
        cut_contents = map(lambda s: list(jieba.cut(s)), text)#得到一个包含分词结果的列表
        if flag == 0:
            cut_contents = [char for char in text]
        stop=open('cn_stopwords.txt', encoding='utf-8')
        stop_words = stop.read().split("\n")# 分割停用词/自定义停用词/关键词
        stpwrdlst = []
        stpwrdlst.extend(list(stop_words))
        contents_new = [] # 定义返回后的结果
        for line in cut_contents: # 遍历处理数据
            line_clean = []
            for word in line:
                if (word in stop_words):  
                    continue
                if check_chinese_type(word):
                    line_clean.append(word)
            contents_new.append(line_clean)
        return contents_new

def data_split(trainFile, testFile):
    trainlist = []
    testlist = []
    for file in fileList:
        f = open(os.path.join(filePath, file), encoding='gb18030')
        if file in bookfile:
            paras = []
            paralist = []
            selected = []
            fullText = f.read()
            paras = fullText.split("\n")
            for i in range(0, len(paras), 5):
                end_index = min(i + 5, len(paras))# 如果是奇数段，直接添加到合并后的段落列表
                paralist.append("\n".join(paras[i:end_index])) # 将当前范围内的段落合并为一个新段落
            paralist = list(filter(filter_para, paralist))
            random.seed(10)
            selected = random.sample(paralist, int((M+N)/len(bookfile)))
            random.shuffle(selected)
            trainlist.extend(selected[0:int(M/len(bookfile))])
            testlist.extend(selected[int(M/len(bookfile)):int((M+N)/len(bookfile))])
    for para in trainlist:
        trainFile.write(para + '\n\n')
    for para in testlist:
        testFile.write(para + '\n\n')
    trainFile.close()
    testFile.close()

def filter_para(s):
    return s and s.strip() and len(s) > 250

def check_chinese_type(words):
    for word in words:
        if word >= u'\u4e00' and word <= u'\u9fa5':
            continue
        else:
            return False
    return True

if __name__ == '__main__':
    M = 1000
    N = 200
    bookfile = ["倚天屠龙记.txt", "笑傲江湖.txt", "天龙八部.txt", "射雕英雄传.txt", "鹿鼎记.txt"]
    filePath = r"book/"  # 文件夹路径
    testFilePath = "test_data.txt"
    trainFilePath = "train_data.txt"

    fileList = os.listdir(filePath)
    open(testFilePath, 'w').close() #如果文件已存在，清空文件的所有内容；如果文件不存在，创建一个空文件
    open(trainFilePath, 'w').close()
    testFile = open(testFilePath, 'a', encoding='utf-8')
    trainFile = open(trainFilePath, 'a', encoding='utf-8')
    data_split(trainFile, testFile)

    flag = 1 #flag为0时以字为单位，为1时以此词单位
    train_data_pre = lda_data(trainFilePath, flag)
    test_data_pre = lda_data(testFilePath, flag)
    token = []
    for t in train_data_pre:
        token.append(len(t))
    for t in test_data_pre:
        token.append(len(t))
    # print(min(token))
    train_data = []
    test_data = []
    token = [5, 10, 20, 40, 50]
    for train in train_data_pre:
        # random.seed(10)
        # train_data.append(random.sample(train, token[4]))
        train_data.append(train)
    for test in test_data_pre:
        # random.seed(10)
        # test_data.append(random.sample(test, token[4]))
        test_data.append(test)
    
    print("训练集段落数:", len(train_data))
    print("测试集段落数:", len(test_data))
    # print(test_data[-1])
    y_train = []
    y_test = []
    for i in range(M):
        y_train.append(i // 200)
    for i in range(N):
        y_test.append(i // 40)

    T = [5, 10, 50, 100, 200, 400] #主题数
    id2word = corpora.Dictionary(train_data + test_data)
    train_texts = train_data
    test_texts = test_data
    train_corpus = [id2word.doc2bow(text) for text in train_texts]
    test_corpus = [id2word.doc2bow(text) for text in test_texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, num_topics=T[2], random_state=100,
                                                update_every=1, chunksize=100, passes=10,
                                                alpha='auto', per_word_topics=True)

    X_train_lda = []
    X_test_lda = []
    for i, item in enumerate(test_corpus):
        topic = lda_model.get_document_topics(item)
        init = np.zeros(M)
        for i, v in topic:
            init[i] = v
        X_test_lda.append(init)

    for i, item in enumerate(train_corpus):
        topic = lda_model.get_document_topics(item)
        init = np.zeros(M)
        for i, v in topic:
            init[i] = v
        X_train_lda.append(init)

    topics = lda_model.print_topics()
    data = []# 准备要写入CSV的数据
    for topic in topics:
        topic_number, topic_data = topic
        topic_keywords = topic_data.split('+')[1:-1]  # 跳过开头的主题编号和最后的空字符串
        for keyword in topic_keywords:
            weight, word = keyword.split('*')
            data.append((topic_number, word.strip(), float(weight)))
    header = ['Topic', 'Keyword', 'Weight']# 将数据写入CSV文件
    with open('lda_topics.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)  # 写入标题行
        for row in data: # 写入主题数据
            csvwriter.writerow(row)
        
    coherence_model_lda = CoherenceModel(model=lda_model, texts=train_data, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Perplexity: ', lda_model.log_perplexity(test_corpus))  # 越低越好
    print('Coherence Score: ', coherence_lda)  # 越高越好

    rf = RandomForestClassifier(n_estimators=300, random_state=10, max_depth=9)
    rf.fit(X_train_lda, y_train)

    # 执行十折交叉验证
    kf = KFold(n_splits=10, shuffle=False)
    scores = cross_val_score(rf, X_train_lda, y_train, cv=kf, scoring='accuracy')
    print("交叉验证分数:", scores)
    print("平均分数:", np.mean(scores))

    predicted_labels = rf.predict(X_test_lda)
    print("实际标签:", y_test)
    print("预测标签:", list(predicted_labels))
    accuracy = accuracy_score(y_test, predicted_labels)
    print(f"模型准确率: {accuracy:.2f}")