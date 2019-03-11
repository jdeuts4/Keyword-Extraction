from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import string
import nltk
from nltk.stem.porter import PorterStemmer

def identity_tokenizer(text):
    return text

def preprocess(document,returnDict=False):
    doc = document.lower()
    doc = ''.join(ch for ch in doc if ch not in string.punctuation)
    tokens = nltk.word_tokenize(doc)
    stemmed = [porter.stem(word) for word in tokens]
    if returnDict:
        return stemmed, dict(zip(stemmed,tokens))
    else:
        return stemmed



    
topKewyords = 5 #top N keywwords to display
    
    
porter = PorterStemmer() #stemming
newsgroups_train = fetch_20newsgroups(remove=('headers','footers'))
data = newsgroups_train.data


#preprocess
text = []
for i in range(len(data)):
    if i%100 == 0:
        print '{:0.2f}%'.format(1.0*i/len(data)*100.0)
    text.append(preprocess(data[i]))
        


# create the transform
vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', 
                             lowercase=False)    
# tokenize and build vocab
vectorizer.fit(text) #or text if preprocess


#example text

d = """Computer programming is the process of designing and building an executable 
computer program for accomplishing a specific computing task. Programming involves 
tasks such as analysis, generating algorithms, profiling algorithms' accuracy and 
resource consumption, and the implementation of algorithms in a chosen programming 
language (commonly referred to as coding[1][2]).The source code of a program is written in
 one or more programming languages. The purpose of programming is to find a 
sequence of instructions that will automate the performance of a task for solving a 
 given problem. The process of programming thus often requires expertise in several 
 different subjects, including knowledge of the application domain, specialized algorithms, 
 and formal logic. Related programming tasks include testing, debugging, maintaining a program's 
source code, implementation of build systems, and management of derived artifacts 
such as machine code of computer programs. These might be considered part of the 
programming process, but often the term software development 
is used for this larger process with the term programming, implementation, 
or coding reserved for the actual writing of source code. Software engineering 
combines engineering
"""




d,unstemmer = preprocess(d,True)

# tf-idf for first document
vector = vectorizer.transform([d])
v = vector.toarray().flatten()

#vocab:tf-idf dictionary
vocab_dict = vectorizer.vocabulary_ # {word: column indices}
keys = vocab_dict.keys()
#convert to ascii
for i in range(len(keys)):
    try:
        keys[i] = str(keys[i])
    except:
        pass
    
vals = vocab_dict.values()
outdict = dict(zip(keys,v[vals]))


#get top N keywords
keywords = sorted(outdict,key=outdict.get,reverse=True)[:topKewyords]
percent = [outdict[i] for i in keywords][:topKewyords]
keywords = [unstemmer[word] for word in keywords] #unstem words for easy reading



#plot
plt.figure(figsize=(7,7))
plt.rcParams.update({'font.size': 15})
y_pos = range(topKewyords)
y_pos.reverse()
plt.barh(y_pos,percent,align='center')
plt.yticks(y_pos,keywords)
plt.ylabel('Keyword')
plt.xlabel('Weight')



