#!usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import random

def map_dict_updown(integer):
    d = {0: '上', 1: '中', 2: '下'}
    return d[integer]
def map_dict_leftright(integer):
    d = {0: '左', 1: '右'}
    return d[integer]
def map_dict_inout(integer):
    d = {0: '内', 1: '中', 2: '外'}
    return d[integer]

def xinyingzengda(value):
    '''
    value: output from model, type: int
    return: type: unicode
    '''
    xinying_keywords = [['心影明显增大', '心影饱满'], ['心影略增大', '心影略大'], ['心影稍大', '心脏稍增大']]

    if value < 0.5:
        return
    elif value < 0.7:
        sel_words = np.random.choice(xinying_keywords[2])
    elif value < 0.9:
        sel_words = np.random.choice(xinying_keywords[1])
    else:
        sel_words = np.random.choice(xinying_keywords[0])
    return sel_words


def feijiejie(position):
    '''
    position: type: list, example: [0,1,1] (0:左右, 1:上中下, 2:内中外)
    return: type: unicode
    '''
    random_value = random.random()
    if position[-1] == 'FeiJieJie':
        feijiejie_keywords = [['{}{}肺野{}带见小结节影', '{}{}肺野{}带见小结节'],
                               ['{}{}肺见小结节影', '{}{}肺见小结节'],
                                ['{}肺{}带见小结节影', '{}肺{}带见小结节影', '{}肺{}带见小结节', '{}肺{}带见小结节'],
                                ['{}肺见小结节影', '{}肺见小结节'],
                                ['{}带见小结节影', '{}带见小结节'],
                                ['心影区域见小结节', '心影区域见结节影', '心影区域见可疑结节', '心影区域见小结节影']]


        if position[0] == 'None' and position[1] == 'None' and position[2] == 'None':
            word = np.random.choice(feijiejie_keywords[5])
        elif position[0] == 'None' and position[1] == 'None':
            word = np.random.choice(feijiejie_keywords[4]).format(map_dict_inout(position[2]))
        elif position[0] == 'None' and position[2] == 'None':
            word = np.random.choice(feijiejie_keywords[3]).format(map_dict_updown(position[1]))
        elif position[1] == 'None' and position[2] == 'None':
            word = np.random.choice(feijiejie_keywords[3]).format(map_dict_leftright(position[0]))
        elif position[0] == 'None':
            word = np.random.choice(feijiejie_keywords[2]).format(map_dict_updown(position[1]), map_dict_inout(position[2]))
        elif position[1] == 'None':
            word = np.random.choice(feijiejie_keywords[2]).format(map_dict_leftright(position[0]), map_dict_inout(position[2]))
        elif position[2] == 'None':
            word = np.random.choice(feijiejie_keywords[1]).format(map_dict_leftright(position[0]), map_dict_updown(position[1]))
        else:
            word = np.random.choice(feijiejie_keywords[0]).format(map_dict_leftright(position[0]), map_dict_updown(position[1]), map_dict_inout(position[2]))

        # if random_value > 0.5:
        #     random_value = random.random()
        #     if random_value > 0.5:
        #         word=word+'可能'
        #     else:
        #         word='考虑'+word+'可能'
        return word
    elif position[-1] == 'FeiJieJie_XinYin_up':
        words = ['心影后方多发高密度结节','心影后方小结节','心影旁可疑结节影','心影后可疑结节影','心影旁小结节','心影处高密度结节',
                 '心影处结节影','心影后方可疑小结节']
        word = random.choice(words)

        return word
    elif position[-1] == 'FeiJieJie_XinYin_down_large_x':
        words = ['左心影重叠处小结节','左心影重叠处高密度结节','左肺下野心影后可疑结节','左肺下野心影旁小结节','左侧心影重叠处结节样密度增高影'
                 '左肺下野心影后小结节影','左下肺野心影后方小结节','左下肺与心影重叠处小结节影','左下肺心影后可疑结节影','左下肺野心影重叠处可疑结节影',
                 '左下肺心影后可疑结节影','左下肺心影旁可疑结节影','心影重叠处结节灶密度增高影','左下肺野心影重叠处小结节影可疑','左肺下野心影重叠区小结节影',
                 '心影重叠处结节影','左下肺野心影后方小结节','左肺下野心影重叠区可疑结节影','左下肺野心影后结节样高密度影','左侧心影后纵隔旁结节影','左肺下野心影重叠区钙化结节']

        word = random.choice(words)
        random_value = random.random()

        return word
    elif position[-1] == 'FeiJieJie_XinYin_down_small_x':
        words = ['右肺下野心影后结节','右肺下野心影后可疑结节','右下肺心影旁可疑结节']
        word = random.choice(words)

        return word
    elif position[-1] == 'FeiJieJie_not_XinYing_Fei':
        words = ['少许结节可能','可疑高密度结节','可疑结节影','可疑小结节影']
        word = random.choice(words)
        return word

def xiongqiangjiye(values):
    '''
    values: type: list, example: [0,1] means [left/no, right/yes]
    return: type: unicode
    '''
    xiongqiangjiye_keywords = np.random.choice(['{}侧胸腔积液', '{}侧胸腔积少量积液考虑', '{}侧胸腔积液可能大'])
    xiongqiangjiye__none_keywords = '胸腔积液可能大'

    if values[0] == 'None' and values[1] == 'None':
        return xiongqiangjiye__none_keywords
    if values[0] == 1 and values[1] == 1:
        if np.random.random() >= 0.5:
            word = xiongqiangjiye_keywords.format('两')
        else:
            word = xiongqiangjiye_keywords.format('双')
    elif values[0] == 0 and values[1] == 1:
        word = xiongqiangjiye_keywords.format('右')
    else:
        word = xiongqiangjiye_keywords.format('左')
    return word


def feiqizhong(value):
    '''
    value: type: bool or 1/0, example: True or 1
    return: type: unicode
    '''
    feiqizhong_keywords = ['肺气肿', '肺气肿可能', '肺气肿改变']
    if value:
        sel_word = np.random.choice(feiqizhong_keywords)
    return sel_word

def HengGe(value):
    words = ['膈面抬高', '横膈稍抬高','横膈稍上抬','横膈上抬']
    if value:
        word = np.random.choice(words)
        if value == 'left':
            word = '左侧' + word
        else:
            word = '右侧' + word
        return word

def XiongMoZH(value):
    words = ['胸膜增厚粘连','胸膜增厚']
    if value:
        word = np.random.choice(words)
        # if value == 'left':
        #     word = '左侧' + word
        # else:
        #     word = '右侧' + word
        return word

def ZhuDongMai(value):
    words = ['主动脉迂曲','主动脉弓迂曲','主动脉稍迂曲增宽','胸主动脉迂曲',]
    if value:
        word = np.random.choice(words)
        return word

def ZDMGaihua(value):
    words = ['主动脉钙化','主动脉结钙化','主动脉结钙化影']
    if value:
        word = np.random.choice(words)
        return word

def jizhucewan(value):
    '''
    value: type: int, example: 9
    return: type: unicode
    '''
    jizhucewan_keywords = [['脊柱稍侧弯', '脊柱略侧弯'], '脊柱轻度侧弯', ['脊柱明显侧弯', '脊柱侧弯畸形']]

    if value < 8:
        sel_word = np.random.choice(jizhucewan_keywords[0])
    elif value < 20:
        sel_word = jizhucewan_keywords[1]
    else:
        sel_word = np.random.choice(jizhucewan_keywords[2])
    return sel_word

def feiwenlizengcu(value):
    '''
    value: type: example: True or 1
    return: type: unicode
    '''
    if value:
        words = ['两肺纹理增多','两肺纹理略多','两肺纹理稍多','两下肺纹理略多']
        word = random.choice(words)
        return word
    else:
        raise ValueError('the input value is not 1 or True')

def shangzonggebk(value):
    '''
    value: type: example: True or 1
    return: type: unicode
    '''
    if value:
        word = '纵隔增宽'
        return word
    else:
        raise ValueError('the input value is not 1 or True')

def feimenzdzn(values):
    '''
    values: type: list, example: [0,1] means [left/no, right/yes]
    return: type: unicode
    '''
    feimenzdzn_keywords = '{}肺门增大增浓'
    feimenzdzn_none_keywords = '肺门增大增浓'

    if values[0] == 'None' and values[1] == 'None':
        return feimenzdzn_none_keywords

    if values[0] == 1 and values[1] == 1:
        word = feimenzdzn_keywords.format('双')
    elif values[0] == 0 and values[1] == 1:
        word = feimenzdzn_keywords.format('右')
    else:
        word = feimenzdzn_keywords.format('左')
    return word


def feizhongkuai(value):
    '''
    position: type: list, example: [0,1] (0:左右, 1:上中下)
    return: type: unicode
    '''
    feizhongkuai_keywords = ['{}{}肺团片影','{}肺团片影']
    feizhongkuai_none_keywords = '肺团片影'

    if value[0] == 'None' and value[1] == 'None':
        return feizhongkuai_none_keywords

    if value[0] == 'None':
        word = feizhongkuai_keywords[1].format(map_dict_updown(value[1]))
    elif value[1] == 'None':
        word = feizhongkuai_keywords[1].format(map_dict_leftright(value[0]))
    else:
        word = feizhongkuai_keywords[0].format(map_dict_leftright(value[0]), map_dict_updown(value[1]))
    return word


def GaiHua(value):
    '''
    position: type: list, example: [0,1] (0:左右, 1:上中下)
    return: type: unicode
    '''
    feigaihua_keywords = ['{}{}肺钙化灶','{}肺钙化灶']
    feigaihua_none_keywords = '肺钙化灶'

    if value[0] == 'None' and value[1] == 'None':
        return feigaihua_none_keywords

    if value[0] == 'None':
        word = feigaihua_keywords[1].format(map_dict_updown(value[1]))
    elif value[1] == 'None':
        word = feigaihua_keywords[1].format(map_dict_leftright(value[0]))
    else:
        word = feigaihua_keywords[0].format(map_dict_leftright(value[0]), map_dict_updown(value[1]))
    return word

def normal(value):
    if value:
        words = ['余两肺未见明显活动性病变','未见活动性病变','余两肺未见活动性病变','未见明确活动性病变','所示肺野未见明显活动性病变','胸片示两肺未见明显活动性病变']
        word = random.choice(words)
        return word
    else:
        raise ValueError('the input value is not 1 or True')


def last_sentence(value):
    '''
    value: type: example: True or 1
    return: type: unicode
    '''
    if value:
        words = ['请结合临床','请结合临床，随访','请结合临床及其它检查，随访','请结合临床及其他相关检查','请结合临床病史及其他相关检查',
                 '请结合临床、病史及其他检查，短期随访复查','请结合临床病史和其它检查随访','请结合临床进一步检查。']
        word = random.choice(words)
        return word
    else:
        raise ValueError('the input value is not 1 or True')

def banpianying(value):
    '''
    position: type: list, example: [0,1] (0:左右, 1:上中下)
    return: type: unicode
    '''
    banpianying_keywords = ['{}{}肺斑片影','{}肺斑片影']
    banpianying_none_keywords = '肺斑片影'

    if value[0] == 'None' and value[1] == 'None':
        return banpianying_none_keywords
    if value[0] == 'None':
        word = banpianying_keywords[1].format(map_dict_updown(value[1]))
    elif value[1] == 'None':
        word = banpianying_keywords[1].format(map_dict_leftright(value[0]))
    else:
        word = banpianying_keywords[0].format(map_dict_leftright(value[0]), map_dict_updown(value[1]))
    return word


def gaihua(value):
    '''
    position: type: list, example: [0,1,0] (0:左右, 1:上中下, 1:主动脉钙化;0:肺钙化)
    return: type: unicode
    '''
    random_value = random.random()
    if value[2] == 'GaiHua_Fei':
        feigaihua_keywords = ['{}{}肺钙化灶','{}肺钙化灶']
        feigaihua_none_keywords = '肺钙化灶'
        if value[0] == 'None' and value[1] == 'None':
            words = ['肺钙化灶','肺钙化灶可能','考虑肺钙化灶可能','肺钙化影','肺钙化影可能',
                     '考虑肺钙化影可能']
            word = random.choice(words)
            return word
        if value[0] == 'None':
            word = feigaihua_keywords[1].format(map_dict_updown(value[1]))
        elif value[1] == 'None':
            word = feigaihua_keywords[1].format(map_dict_leftright(value[0]))
        else:
            word = feigaihua_keywords[0].format(map_dict_leftright(value[0]), map_dict_updown(value[1]))


        return word
    elif value[2] == 'GaiHua_XinYin_up':
        words = ['主动脉迂曲钙化','主动脉结突出伴钙化','主动脉迂曲伴壁钙化','主动脉结钙化',
                 '主动脉结钙化影']
        word = random.choice(words)

        return word
    elif value[2] == 'GaiHua_XinYin_down_large_x':
        words = ['心影重叠区团块状高密度钙化影','左心影区见点状钙化灶','左心影区见团块状高密度钙化影','心影重叠区团见点状钙化灶',
                 '左肺下野心影重叠区钙化','左肺下野心影重叠区见点状钙化灶','左肺下野心影重叠区见高密度钙化影']
        word = random.choice(words)
        random_value = random.random()

        return word
    elif value[2] == 'GaiHua_XinYin_down_small_x':
        words = ['心影区见点状钙化灶','心影区见钙化影','心影区见钙化','心影区团块状高密度钙化影',
                 '右心影区见点状钙化灶','右肺下野心影重叠区见点状钙化灶','右心影区团块状高密度钙化影']
        word = random.choice(words)

        return word
    elif value[2] == 'GaiHua_not_XinYin_Fei':
        words = ['少许钙化可能','部分钙化','主动脉迂曲伴壁钙化','主动脉结钙化','考虑钙化影可能']
        word = random.choice(words)
        return word


def RuTouYing(value):
    '''
    position: type: string, example: 'left', 'right'
    return: type: unicode
    '''
    assert value in ['left', 'right']
    if 'left' == value:
        return '左下肺可疑乳头影'
    else:
        return '右下肺可疑乳头影'

if __name__=='__main__':
    # a = gaihua([1,0,0])
    # print(a)
    # a = banpianying(['None', 'None'])
    # print(a)
    # a = xueguanbigaihua(True)
    # print(a)
    # a = feigaihua(['None', 'None'])
    # print(a)
    # a = feizhongkuai(['None','None'])
    # print(a)
    # a = feimenzdzn(['None','None'])
    # print(a)
    # a = shangzonggebk(True)
    # print(a)

    a = RuTouYing('right')
    print(a)
    # a = xinyingzengda(0.98)
    # print(a)
    # print(type(a))
    # print('-'*20)
    # a = feijiejie([1,2,'None'])
    # print(a)
    # a = feijiejie([0,0,0])
    # print(type(a))
    # print('-'*20)
    # a = xiongqiangjiye(['None','None'])
    # print(a)
    # print('-'*20)
    # a = feiqizhong(1)
    # print(a)
    # print('-'*20)
    # a = jizhucewan(13)
    # print(a)
