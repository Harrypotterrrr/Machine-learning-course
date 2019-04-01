# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 17:30:50 2019

@author: 49387
"""
import os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
#获取文件夹里的所有文件名
def get_file_name(file_dir):   
    filename=[]
    for root, dirs, files in os.walk(file_dir):
        filename=files
    return filename
def clean_txt(text):
       text = BeautifulSoup(text, 'html.parser').get_text()    #提取正文
       text = re.sub(r'[^a-zA-Z]', ' ', text)    #清理特殊符号
       words = text.lower().split()     #空格分来
       words = [w for w in words if w not in stopwords.words('english')]    #清理停顿词
       return ' '.join(words)

#读取所有的句子存成网站的表格形式
def Alltxt2csv():
    NLPform={}
    NLPform['id']=[]
    NLPform['sentiment']=[]
    NLPform['review']=[]
    file_path_neg=os.path.join(os.getcwd(),r'aclImdb\test\neg')
    file_path_pos=os.path.join(os.getcwd(),r'aclImdb\test\pos')
    file_name_neg=get_file_name(file_path_neg)
    file_name_pos=get_file_name(file_path_pos)
    for file in file_name_neg:
        print(os.path.join(file_path_neg,file))
        with open(os.path.join(file_path_neg,file),'rb') as piece:
            NLPform['id'].append(file.strip('.txt'))
            NLPform['sentiment'].append(0)
            NLPform['review'].append(clean_txt(piece.read()))
        piece.close()
        
    for file in file_name_pos:
        print(os.path.join(file_path_neg,file))
        with open(os.path.join(file_path_pos,file),'rb') as piece:
            NLPform['id'].append(file.strip('.txt'))
            NLPform['sentiment'].append(1)
                
            NLPform['review'].append(clean_txt(piece.read()))
            
        piece.close()
    df=pd.DataFrame(NLPform)
    df.to_csv('reviewsTest.csv',sep=',')
         
                  
    
Alltxt2csv()
#nltk.download()
