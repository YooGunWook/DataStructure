#3. 컬랙션 자료구조 

# 컬렉션은 시퀀스와 다르게 데이터를 서로 연관시키지 않고 모아두는 컨테이너. 
# 컬렉션은 세가지 속성을 가지고 있다.
# 1. 멤버십 연산자: in
# 2. 크기 함수: len(seq)
# 3. 반복성 : 반복문의 데이터를 순회한다. 
# 
# 3.1 셋
# 
# 셋은 반복 가능하고, 가변적이며, 중복 요소가 없고, 정렬되지 않은 컬렉션 데이터 타입. 
# 셋은 멤버십 테스트 및 중복 항목 제거에 사용된다.  
# 시간 복잡도는 O(1)이다 
# 합집합은 O(m+n), 교집합은 O(n)이다.

# 3.1.1 셋 메서드

# add()
people = {'버피','에인절','자일스'}
people.add('윌로')
print(people) # {'자일스', '윌로', '버피', '에인절'}

# update()와 |= 연산자 (둘이 같은 method.)

people = {'버피','에인절','자일스'}
people.update({'로미오','줄리엣','에인절'})
print(people) # {'로미오', '버피', '줄리엣', '자일스', '에인절'}
people |= {'리키','유진'}
print(people) # {'로미오', '버피', '줄리엣', '자일스', '에인절', '유진', '리키'}

# union(), | -> 합집합
people = {'버피','에인절','자일스'}
print(people.union({'로미오','줄리엣'})) # {'로미오', '자일스', '에인절', '버피', '줄리엣'}
print(people|{'브라이언'}) # {'에인절', '버피', '브라이언', '자일스'}

# intersection()과 & 연산자 -> 교집합
people = {'버피','에인절','자일스','이안'}
vampires = {'에인절','자일스','윌로'}
print(people.intersection(vampires)) # {'에인절', '자일스'}
print(people & vampires) # {'에인절', '자일스'}

# difference()와 - 연산자 -> 차집합
people = {'버피','에인절','자일스','이안'}
vampires = {'에인절','자일스','윌로'}
print(people.difference(vampires)) # {'버피', '이안'}
print(people-vampires)

# clear()
people = {'버피','에인절','자일스','이안'}
people.clear()
print(people) # set()

# discard(), remove(), pop()
# discard(x)는 x를 제거하며 반환값 없음. remove(x)는 discard와 같지만, x가 없으면 에러남. pop(x)는 x를 반환하고 제거된다. 그리고 x가 없으면 에러남. 
countries = {'프랑스','스페인','영국'}
print(countries.discard('한국')) # 에러 안남. 
print(countries.remove('일본')) # 에러 남. 
print(countries.pop()) # 무작위로 빠짐.
print(countries.discard('스페인'))
print(countries.remove('영국'))
print(countries.pop()) # 에러남 

# 3.1.2 셋과 리스트

# 리스트 타입은 셋 타입으로 변환 가능함. 

def remove_dup(l1):
    return list(set(l1)) # 중복된 항목 제거

def intersection(l1,l2):
    return list(set(l1) & set(l2)) # 교집합 결과 반환

def union(l1,l2):
    return list(set(l1) | set(l2))

def test_sets_operations_with_lists():
    l1 = [1,2,3,4,5,5,9,11,11,15]
    l2 = [4,5,6,7,8]
    l3 = []
    assert(remove_dup(l1) == [1,2,3,4,5,9,11,15])
    assert(intersection(l1,l2) == [4,5])
    assert(union(l1,l2) == [1,2,3,4,5,6,7,8,9,11,15])
    assert(remove_dup(l3) == [])
    assert(intersection(l3,l2) == l3)
    assert(sorted(union(l3,l2)) == sorted(l2))
    print('테스트 통과')

if __name__ == '__main__':
    test_sets_operations_with_lists()

# 딕셔너리에서도 셋 옵션을 사용할 수 있다. 

def set_operations_with_dict():
    pairs = [ ('a',1), ('b',2), ('c',3)]
    d1 = dict(pairs)
    print('딕셔너리1\t: {0}'.format(d1))

    d2 = {'a':1, 'c':2, 'd':3,'e':4}
    print('딕셔너리2\t: {0}'.format(d2))

    intersection_items = d1.keys() & d2.keys()
    print('d1 n d2 (키)\t: {0}'.format(intersection_items)

    subtraction1 = d1.keys() - d2.keys()
    print('d1 - d2 (키)\t: {0}'.format(subtraction1))

    subtraction2 = d2.keys() - d1.keys()
    print('d2 - d1 (키)\t: {0}'.format(subtraction2))

    subtraction_items = d1.items() - d2.items()
    print('d1 - d2 (키,값)\t: {0}'.format(subtraction_items))

    # 딕셔너리의 특정 키를 제외한다. 

    d3 = {key: d2[key] for key in d2.keys()-{'c','d'}}
    print('d2-{{c,d}}\t: {0}'.format(d3))

if __name__ == '__main__':
    set_operations_with_dict()


# 3.2 딕셔너리

# 해시 테이블로 구성되어 있다. 해시 함수는 특정 객체에 해당하는 임의의 정수 값을 상수 시간 내에 계산한다. 
# 연관 배열의 인덱스로 사용된다. 

print(hash(42)) # 42
print(hash('hello')) # -4740507532167689545

# 컬렉션 매핍 타입인 딕셔너리는 반복 가능함. 그리고 맴버십 in 연산자와 len()함수도 지원해줌. 
# 매핑은 키값 컬렉션.

# 딕셔너리의 항목은 고유하므로, 항목에 접근하는 시간복잡도는 O(1)이다.
# 항목의 추가, 제거 가능. 
# 삽입 순서는 기억하지 않는다. 따라서 인덱스 위치를 사용할 수 없다. 

tarantino = {}
tarantino['name'] ='쿠엔틴 타란티노'
tarantino['job'] = '감독'
print(tarantino) # {'name': '쿠엔틴 타란티노', 'job': '감독'}

sunnydale = dict({'name':'버피','age':16,'hobby':'게임'})
print(sunnydale) # {'name': '버피', 'age': 16, 'hobby': '게임'}

sunnydale = dict(name = '자일스',age = 20,hobby = '게임')
print(sunnydale) # {'name': '자일스', 'age': 20, 'hobby': '게임'}

sunnydale = dict([('name','건욱'),('age',20),('hobby','개발')])
print(sunnydale) # {'name': '건욱', 'age': 20, 'hobby': '개발'}

# 3.2.1 딕셔너리 메서드

# setdefault()
# 키의 존재 여부를 모른 채 접근할 때 사용된다. 
# A.setdefault(key, default) A에 key가 존재할 경우 키에 해당하는 값을 얻을 수 있고, key가 존재하지 않으면, 새 키와 기본값 default가 딕셔너리에 저장된다. 

def usual_dict(dict_data):
    # 일반 방식
    newdata = {}
    for k,v in dict_data:
        if k in newdata:
            newdata[k].append(v)
        else:
            newdata[k] = [v]
    return newdata

def setdefault_dict(dict_data):
    # setdefault()
    newdata = {}
    for k, v in dict_data:
        newdata.setdefault(k,[]).append(v)
    return newdata

def test_setdef():
    dict_data = (
                ('key1','value1'),
                ('key1','value2'),
                ('key2','value3'),
                ('key2','value4'),
                ('key2','value5'),
                )
    print(usual_dict(dict_data))
    print(setdefault_dict(dict_data))

if __name__ == '__main__':
    test_setdef()

# {'key1': ['value1', 'value2'], 'key2': ['value3', 'value4', 'value5']}
# {'key1': ['value1', 'value2'], 'key2': ['value3', 'value4', 'value5']}
# 같은 값이 나오는 것을 확인할 수 있다. 

# A.update(B)는 딕셔너리 A에 딕셔너리 B의 키가 존재하면, 기존 A의 (키,값)을 B의 (키,값)으로 갱신한다. B에 A의 키값이 존재하지 않으면 B의 키,값을 A에 추가한다. 

d = {'a': 1 , 'b' : 2}
d.update({'b':10})
print(d) # {'a': 1, 'b': 10}
d.update({'c':8})
print(d) # {'a': 1, 'b': 10, 'c': 8}

# get()은 딕셔너리 A의 키값을 반환한다. key가 없으면 반환하지 않음. 

sunnydale = dict(name = '건욱', age = 17, hobby = '게임')
print(sunnydale.get('hobby')) # 게임
print(sunnydale['hobby']) # 게임
print(sunnydale.get('hello')) # None
sunnydale['hello'] # Error

# items(), values(), keys()는 딕셔너리 뷰. 딕셔너리의 항목을 조회하는 읽기 전용의 반복 가능한 객체. 
sunnydale = dict(name = '건욱', age = 17, hobby = '게임')
print(sunnydale.items()) # dict_items([('name', '건욱'), ('age', 17), ('hobby', '게임')])
print(sunnydale.values()) # dict_values(['건욱', 17, '게임'])
print(sunnydale.keys()) # dict_keys(['name', 'age', 'hobby'])

sunnydale_copy = sunnydale.items()
#sunnydale_copy['address'] = '서울' # 에러남
sunnydale['address'] = '서울'
print(sunnydale) # {'name': '건욱', 'age': 17, 'hobby': '게임', 'address': '서울'}

# pop(), popitem()
# A.pop(key)는 딕셔너리 A의 key 항목을 제거한 후 그 값을 반환함. A.popitem()은 A에서 (키,값)을 제거한 후, (키, 값)을 반환한다. 

# clear()는 딕셔너리의 모든 항목을 제거함. 

# 3.2.2 딕셔너리 성능 측정
# 멤버십 연산에 대한 시간복잡도는 리스트의 경우 O(n)이고 딕셔너리는 O(1)이다. 

# 3.2.3 딕셔너리 순회
# 딕셔너리도 sorted() 함수를 사용할 수 있다. 

# 3.2.4 딕셔너리 분기
def hello():
    print('hello')

def world():
    print('world')

action = 'h'

functions = dict(h = hello, w = world)
print(functions[action]()) # hello # None

# 3.3 파이썬 컬렉션 데이터 타입

# 3.3.1 기본 딕셔너리
# 기본 딕셔너리는 collections.defaultdict 모듈에서 제공하는 추가 딕셔너리 타입임. 
# 내장 딕셔너리의 모든 연산자와 메서드를 사용할 수 있고, 추가로 다음 코드와 같이 누락된 키도 처리할 수 있음. 

from collections import defaultdict

def defaultdict_example():
    pairs = {('a',1),('b',2),('c',3)}

    # 일반 딕셔너리
    d1 = {}
    for key, value in pairs:
        if key not in d1:
            d1[key] = []
        d1[key].append(value)
    print(d1)

    # defaultdict
    d2 = defaultdict(list)
    for key, value in pairs:
        d2[key].append(value)
    print(d2)

if __name__ == '__main__':
    defaultdict_example()
    # {'b': [2], 'a': [1], 'c': [3]}
    # defaultdict(<class 'list'>, {'b': [2], 'a': [1], 'c': [3]})

# 3.3.2 정렬된 딕셔너리 
# 정렬된 딕셔너리는 collections.OrderedDict 모듈에서 처리 가능. 
# 내장 딕셔너리의 모든 연산자와 메서드를 사용할 수 있고, 추가로 다음 코드와 같이 누락된 키도 처리할 수 있음.

from collections import OrderedDict
tasks = OrderedDict()
tasks[8031] = '백업'
tasks[4027] = '이메일 스캔'
tasks[5733] = '시스템 빌드'
print(tasks)

# OrderedDict([(8031, '백업'), (4027, '이메일 스캔'), (5733, '시스템 빌드')])

from collections import OrderedDict

def OrderedDict_example():
    pairs = [('c',1),('b',2),('a',3)]

    # 일반 딕셔너리
    d1 = {}
    for key, value in pairs:
        if key not in d1:
            d1[key] = []
        d1[key].append(value)
    for key in d1:
        print(key, d1[key])

    # OrderedDict
    d2 = OrderedDict(pairs)
    for key in d2: 
        print(key, d2[key])


if __name__ == '__main__':
    OrderedDict_example()

    # c [1]
    # b [2]
    # a [3]
    # c 1
    # b 2
    # a 3

# 키 값을 변경해도 순서는 변하지 않는다. 맨끝으로 저장하기 위해서는 해당 항목을 삭제한 후 다시 삽입해야한다. popitem()을 호출해서 사용해도 된다. 
# 현재 python 버전은 3.7이기 때문에 따로 OrderedDict를 해주지 않아도 순서가 보존된다. 

# 3.3.3 카운터 딕셔너리
# 카운터 타입은 해시 가능한 객체를 카운팅하는 특화된 서브클래스. collections.Counter 모듈에서 제공함.

from collections import Counter

def counter_example():
    # 항목의 발생 횟수를 매핑하는 딕셔너리를 생성한다. 
    seq1 = [1,2,3,5,1,2,5,5,2,5,1,4]
    seq_counts = Counter(seq1)
    print(seq_counts)

    # 항목의 발생 횟수를 수동으로 갱신하거나, update() 메서드를 사용할 수 있다. 
    seq2 = [1,2,3]
    seq_counts.update(seq2)
    print(seq_counts)

    seq3 = [1,4,3]
    for key in seq3:
        seq_counts[key] += 1
    print(seq_counts)

    # a+b, a-b 같은 셋 연산을 사용할 수 있다.
    seq_counts_2 = Counter(seq3)
    print(seq_counts_2)
    print(seq_counts + seq_counts_2)
    print(seq_counts - seq_counts_2)

if __name__ == '__main__':
    counter_example()

# Counter({5: 4, 1: 3, 2: 3, 3: 1, 4: 1})
# Counter({1: 4, 2: 4, 5: 4, 3: 2, 4: 1})
# Counter({1: 5, 2: 4, 5: 4, 3: 3, 4: 2})
# Counter({1: 1, 4: 1, 3: 1})
# Counter({1: 6, 2: 4, 3: 4, 5: 4, 4: 3})
# Counter({1: 4, 2: 4, 5: 4, 3: 2, 4: 1})

# 3.4 연습문제 

# 3.4.1 단어 횟수 세기

# collections.Counters의 most_common()을 사용하면 문자열에서 가장 많이 나오는 단어와 횟수를 구할 수 있음.

from collections import Counter

def find_top_N_recuring_words(seq,N):
    dcounter = Counter()
    for word in seq.split():
        dcounter[word] += 1
    return dcounter.most_common(N)

def test_find_top_N_recuring_words():
    seq = '버피 에인절 몬스터 젠더 월로 버피 몬스터 슈퍼 버피 에인절'
    N = 3
    assert(find_top_N_recuring_words(seq,N) == 
        [('버피',3),('에인절',2),('몬스터',2)])
    print('테스트 통과')

if __name__ == '__main__':
    test_find_top_N_recuring_words()

# 3.4.2 에너그램

# 문장 또는 단어의 철자 순서를 바꾸는 놀이. 
# 두 문자열이 서로 애너그램인지 확인하고 싶다고 할 때, 셋은 항목의 발생 횟수를 계산하지 않고, 리스트의 항목을 정렬하는 시간복잡도는 최소 O(nlogn)이다. 
# 따라서 딕셔너리를 사용하는 것이 가장 효율적일 수 있다. 

from collections import Counter

def is_anagram(s1, s2):
    counter = Counter()
    for c in s1:
        counter[c] += 1
    for c in s2:
        counter[c] -= 1
    for i in counter.values():
        if i:
            return False
    return True

def test_is_anagram():
    s1 = 'marina'
    s2 = 'aniram'
    assert(is_anagram(s1,s2) is True)
    s1 = 'google'
    s2 = 'gouglo'
    assert(is_anagram(s1,s2) is False)
    print('테스트 통과')

if __name__ == '__main__':
    test_is_anagram()

# 두 문자열이 애너그램인지 확인하는 또 다른 방법은 해시 함수의 속성을 이용하는 것임.
# ord()함수는 인수가 유니코드 객체일 때, 문자의 유니코드를 나타내는 정수를 반환한다.
# 인수가 8비트 문자열인 경우 바이트 값을 반환한다. 
# 모든 문자의 ord() 함수 결과를 더했을 때, 반환값이 같으면 두 문자열은 애너그램. 

import string

def hash_func(astring):
    s = 0
    for one in astring:
        if one in string.whitespace: # string.whitespace는 띄어쓰기나 빈공간이라고 생각하자. 그래서 여기서 빈공간이면 continue문으로 pass할 수 있게 했다. 
            continue
        s = s + ord(one)
    return s

def find_anagram_hash_function(word1, word2):
    return hash_func(word1) == hash_func(word2)

def test_find_anagram_hash_function():
    word1 = 'buffy'
    word2 = 'bffyu'
    word3 = 'bffya'
    assert(find_anagram_hash_function(word1,word2) is True)
    assert(find_anagram_hash_function(word1,word3) is False)
    print('테스트 통과')

if __name__ =='__main__':
    test_find_anagram_hash_function()

# 3.4.3 주사위 합계 경로

from collections import Counter, defaultdict

def find_dice_probabilities(S, n_faces = 6):
    if S > 2 * n_faces or S < 2:
        return None

    cdict = Counter()
    ddict = defaultdict(list)

    for dice1 in range(1, n_faces+1):
        for dice2 in range(1, n_faces+1):
            t = [dice1, dice2]
            cdict[dice1+dice2] += 1
            ddict[dice1+dice2].append(t)

    return [cdict[S],ddict[S]]

def test_find_dice_probabilities():
    n_faces = 6
    S = 5
    results = find_dice_probabilities(S,n_faces)
    print(results)
    assert(results[0] == len(results[1]))
    print('테스트 통과')

if __name__ == '__main__':
    test_find_dice_probabilities()

# 3.4.4 단어의 중복 문자 제거 

import string

def delete_unique_word(str1):
    table_c = {key : 0 for key in string.ascii_lowercase} # 소문자 모두 dict값으로 변환
    for i in str1:
        table_c[i] += 1
    for key, value in table_c.items():
        if value > 1:
            str1 = str1.replace(key,'')
    return str1

def test_delete_unique_word():
    str1 = 'google'
    assert(delete_unique_word(str1) == 'le')
    print('테스트 통과')

if __name__ == '__main__':
    test_delete_unique_word()

