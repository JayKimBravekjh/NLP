import sqlite3
import pandas as pd

c = sqlite3.connect("database.sqlite")
ds = pd.read_sql("select Text", c)
print(ds.head())


## exploration of a few top words
from sklearn.feature_extraction import text
import nltk
import re
stop_ = list(text.ENGLISH_STOP_WORDS)
d = { }
for i in range(len(ds.PaperText)):
    s = ds.PaperText[i].lower()
    s = s.replace("\n", " ")
    s = re.sub("[^a-z]", " ", s)
    s = s.replace("  ", " ")
    a = set(s.split(" "))
    for b_ in a:
        if b_ not in stop_ and len(b_)>1:
            if b_ not in d:
                d[b_] = 1
            else: 
                d[b_]+= 1
ds2 = pd.DataFrame.from_dict(d, orient='index')
ds2.columns = ['count'] 
ds2 = ds2.sort_values(by=['count'], ascending=[False])
t_=" "
for i in range(100):
    t_ += ds2.index[i] + "[" + str(ds2["count"][i]) + "], " 
print (t_)


## top words being used most frequently
for x in range(5):
    z = { } 
    t_ = ds2.index[x]
    print(t_)
    for i in range(len(ds.PaperText)):
        s = ds.PaperText[i].replace("\n", " ")
        s = re.sub("[^a-zA-Z]", " ", s)
        a = s.replace("  ", " ").split(" ")
        for b_ in range(len(a)):
            if str(a[b_].lower()).find(t_)>0 or str(a[b_].lower()) == t_:
                ss_=""
                se_=""
                if b_-2 >= 0:
                    ss_ = (" ").join([a[b_-2], a[b_-1]])
                if b_+2 < len(a):
                    se_ = (" ").join([a[b_+1], a[b_+2]])
                s_ = (" ").join([ss_,str(a[b_]), se_])
                if s_ not in z:
                    z[s_] = 1
                else:
                    z[s_]+=1
    ds3 = pd.DataFrame.from_dict(z, orient='index')
    ds3.columns = ['count']
    ds3 = ds3.sort_values(by=['count'], ascending=[False])
    print(ds3.head())
    
  
    ## most unique and most generic word papers.
    from IPython.display import HTML
    ds5 = pd.read_sql("select PdfName, PaperText from Papers", c)
    u = {}
    bm_ = 0
    for b_ in d:
        bm_ += int(d[b_])
    
    for i in range(len(ds5.PaperText)):
        r_ = 0.0
        s = ds5.PaperText[i].replace("\n"," ")
        s = s.lower()
        s = re.sub("[^a-z]", " ", s)
        a = list(set(s.replace("  ", " ").split(" ")))
        for b_ in range(len(a)):
            if a[b_] in d:
                r_+=d[a[b_]]/bm_
        u[ds5.PdName[i]]=r_
    
    ds6 = pd.DataFrame.from_dict(u, orient='index')
    ds6.columns = ['uu']
    ds6 = ds6.sort_values(by=['un'], ascending=[True])
    te_ = "<strong>Most Unique to Most Generic Wording:</strong><br><u1>"
    for i in range(len(ds6.un)):
        te_ += "<li><a href='https://papers.nips.cc/paper/" + ds6.index[i] + "'>" + ds6.index[i] + "</a> [" + str(ds6.un[i]) + "]</>[" + str(ds6.un[i]) + "]</li><br>"
    HTML(te_+"</ul>")
    
## identification of all the NN Math Formulas from the PDF files. 

ds_m = pd.read_sql("select PaperText from Papers where PdfName = '5816-evaluating-the-statistical-significance-of-biclusters.pdf'", c)
print(ds_m.PaperText[0][ds_m.PaperText[0].find("Without loss of generality,"):ds_m.PaperText[0].find("For our theoretical results,")])

## import pip
m_ = sorted([i.key for i in pip.get_installed_distributions()])
print(m_)
for mo_ in m_:
    if str(mo_.lower()).find("pdf") > 0:
        print(mo_)

## testing some nodes
import matplotlib.pyplot as plt
import networkx as n
from operator import itemgetter
import itertools
z={}
t_ = "model"
for i in range(len(ds.PaperText)):
    s = ds.PaperText[i].replace("\n"," ")
    s = s.lower()
    s = re.sub("[^a-z]", " ", s)
    a = s.replace("  ", " ").split(" ")
    a = [word for word in a if word not in stop_ and len(word) > 3]
    for b_ in range(len(a)):
        if str(a[b_].lower()).find(t_) > 0 or str(a[b_].lower()) == t_:
            if b_-3>=0:
                l1=b_ - 3
            else: 
                l1=0 
            if b_ +1 < len(a):
                h1 = b_ + 1
            else:
                h1 = len(a) - 1
            for j in range(l1, h1):
                if (a[j]+"|"+a[j+1]) not in z:
                    z[a[j]+"|" + a[j+1] = 1
                else:
                    z[a[j]+"|"+a[j+1] += 1
a = [ ] 
for z_ in z: 
    a.append([z_.split("|")[0], z_.split("|")[1], z[z_]])
a = sorted(a, key=itemgetter(1))

# networkx Graph
g = n.Graph()
g.clear()
for i in range(len(a)):
    g.add_edge(a[i][0], a[i][1])
pos = n.spring_layout(g)
plt.figure(figsize=(12, 12))
n.draw(g)
n.draw_networkx_labels(g, pos, font_size=9, font_family='sans-serif')
plt.savefig('models.png')

    
    
    
