#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:56:07 2018
@author  : weikaiqi
Function : clean text
"""

import re
import pandas as pd
from tqdm import tqdm
from googletrans import Translator


def reduce_lengthening(text):
    '''
        remove repeated characters
        '''
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def replace_time_year(text):
    '''
        replace time and year
    '''
    text = re.sub(r"\d{1,2}:\d{1,2}", ' time ', text)
    text = re.sub(r"\d{1,2}:\d{1,2}:\d{1,2}", ' time ', text)
    text = re.sub(r"\d{4}", ' year ', text)
    return text

def replace_ip(text):
    '''
        replace ip address
    '''
    text = re.sub(r"\d+\.\d+\.\d+\.\d+", "ip", text)
    return text

def replace_number(text):
    '''
        replace number (if n_word smaller than 100)
    '''
    if len(text.split()) < 100:
        text = text.replace('\d', ' number ')
    return text

def remove_repeat_word(text):
    '''
        remove repeated word
    '''
    text = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', text)
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    return text

translator = Translator()
zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')

def tran_en(text):
    '''
        tanslate chinese to english
    '''
    if zh_pattern.search(text):
        text = translator.translate(text,dest='en').text
    return text

def standardize_df(df, text_field):
    
    df[text_field] = df[text_field].str.replace(r"http\S+", " url ")
    df[text_field] = df[text_field].str.replace(r"http", " url ")
    df[text_field] = df[text_field].str.replace(r"@\S+", " email ")
    df[text_field] = df[text_field].str.replace(r"@", " at ")
    df[text_field] = df[text_field].str.replace(r"Image:\S+", " image ")
    df[text_field] = df[text_field].str.replace(r"tags#\S+", " tags ")
    df[text_field] = df[text_field].str.replace(r"Wikipedia_talk\S+", " ")
    df[text_field] = df[text_field].str.lower()
    
    df[text_field] = df[text_field].apply(lambda x: replace_ip(x))
    df[text_field] = df[text_field].apply(lambda x: replace_time_year(x))
    df[text_field] = df[text_field].apply(lambda x: replace_number(x))
    
    df[text_field] = df[text_field].str.replace("what's", "what is ")
    df[text_field] = df[text_field].str.replace("\'s", " ")
    df[text_field] = df[text_field].str.replace('\'ve', " have ")
    df[text_field] = df[text_field].str.replace('don\'t', ' do not ')
    df[text_field] = df[text_field].str.replace('can\'t', ' can not ')
    df[text_field] = df[text_field].str.replace('n\'t', " not ")
    df[text_field] = df[text_field].str.replace('i\'m', "i am ")
    df[text_field] = df[text_field].str.replace(r"\'re", " are ")
    df[text_field] = df[text_field].str.replace(r'\'d', " would ")
    df[text_field] = df[text_field].str.replace(r'\'ll', " will ")
    df[text_field] = df[text_field].str.replace(r' a ', " ")
    df[text_field] = df[text_field].str.replace(r' the ', " ")
    df[text_field] = df[text_field].str.replace(r' an ', " ")
    df[text_field] = df[text_field].str.replace(r' at ', " ")
    df[text_field] = df[text_field].str.replace(r' to ', " ")
    df[text_field] = df[text_field].str.replace(r' or ', " ")
    df[text_field] = df[text_field].str.replace(r' on ', " ")
    df[text_field] = df[text_field].str.replace(r' in ', " ")
    
    df[text_field] = df[text_field].str.replace(r"谢谢", " Thanks ")
    df[text_field] = df[text_field].str.replace(r"很好", " Great ")
    df[text_field] = df[text_field].str.replace(r"你好", " Hello ")
    df[text_field] = df[text_field].str.replace(r"您好", " Hello ")
    df[text_field] = df[text_field].str.replace(r"屌你老母", " motherfucker ")
    df[text_field] = df[text_field].str.replace(r"純血主義", " Korean ethnic nationalism ")
    
    df[text_field] = df[text_field].str.replace(r"肏", " fuck ")
    df[text_field] = df[text_field].str.replace(r"操", " fuck ")
    df[text_field] = df[text_field].str.replace(r"激情", " porn ")
    df[text_field] = df[text_field].str.replace(r"视频", " video ")
    df[text_field] = df[text_field].str.replace(r"呆B", " bitch ")
    df[text_field] = df[text_field].str.replace(r"屌", " fuck ")
    df[text_field] = df[text_field].str.replace(r"小姐", " prostitute ")
    df[text_field] = df[text_field].str.replace(r"𨳒你老母个閪", " fuck your mother ")
    df[text_field] = df[text_field].str.replace(r"臭閪生了你这个臭仔", " son of bitch ")
    df[text_field] = df[text_field].str.replace(r"屌你老母閪", " fuck your mother ")
    df[text_field] = df[text_field].str.replace(r"臭妈的烂B", " fuck your mother ")
    
    
    df[text_field] = df[text_field].str.replace('[,="\.\]\[\?\!\/\'\(\)\-|\\\——]+',' ')
    df[text_field] = df[text_field].str.replace('[\+\:;•¢「」¤><#→™]+',' ')
    df[text_field] = df[text_field].str.replace('[\_·✰@“”’‘☎☓%\{\}~]+',' ')
    df[text_field] = df[text_field].str.replace('[♥♠♦♣⋅£☼&´♫₪–°☏]+',' ')
    df[text_field] = df[text_field].str.replace('[\^¿✉←ⓣ˜≈«»☺❤☯➥…¨]+',' ')
    df[text_field] = df[text_field].str.replace('[≠≤（）《》`№©♪♬♩✍¡►®§√∞⁄]+',' ')
    df[text_field] = df[text_field].str.replace('[△⟨⟩∇∆⇔¦⇒└┐æ🗽☿？₰≡╟☥✄−─╢‖]+',' ')
    
    df[text_field] = df[text_field].str.replace('[🎄🍁☘🍌✎✐●✓★☆☀×◕‿◕♂⁂]+',' ')
    df[text_field] = df[text_field].str.replace('[😂😄😊😢😃😉😅😜😏]+',' ')
    df[text_field] = df[text_field].str.replace('[☮☸♑☢☣🙈🙉🙊̇☄☽◯☾✋🏼⚇♔💩]+',' ')
    df[text_field] = df[text_field].str.replace('[👍☠😀😔💜⁠۩۞✔♝♚�☛➔⊕⊗]+',' ')
    df[text_field] = df[text_field].str.replace('[🎤✈➨¬➪‡↔↑↓¶✆☑◀▶📞📧✿▲╦╩✽∅]+',' ')
    df[text_field] = df[text_field].str.replace('[☭☻｡¸¯߷♀☃⚔◄☝💬━ºø]+',' ')
    df[text_field] = df[text_field].str.replace('[½⅓⅔¼¾⅛⅜⅝⅞ョ℠∙：▎۝҈♨！☞☜‽┘┌]+',' ')
    df[text_field] = df[text_field].str.replace('[≼≽█ロ◥๛〈⌊〉✤┏┓┃╔═╗║╚╝┗┛↗‹›✭ⓐ㊟]+',' ')
    df[text_field] = df[text_field].str.replace('[○▾↨↕☤➜֑❝❞‑ोǀǃǂǁ¹²³⁴⁰±▪₂✘├⇝ٔ✒ʼ❉│‒︵❦❖✗✫⁞『』آ╫ŧ㎥■【】]+',' ')
    df[text_field] = df[text_field].str.replace('[†*$]+',' ')

    df[text_field] = df[text_field].apply(lambda x: tran_en(x))

    df[text_field] = df[text_field].str.replace(r"[^A-Za-z]", " ")
    df[text_field] = df[text_field].apply(lambda x: reduce_lengthening(x))
    df[text_field] = df[text_field].apply(lambda x: remove_repeat_word(x))
    
    return df

def CleanDataText(df, textField):
    df.fillna('_na_')
    df = standardize_df(df, textField)
    return df
        

