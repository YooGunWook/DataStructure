# chapter 2 내장 시퀀스 타입

# 시퀀스 타입은 4가지 속성을 가지고 있다. 
# 맴버십 연산: in 키워드 사용
# 크기 함수: len(seq)
# 슬라이싱 속성: seq[:-1]
# 반복성: 반복문에 있는 데이터를 순회할 수 있다. 

# python에는 문자열, 튜플, 리스트, 바이트 배열, 바이트 등 5개의 내장 시퀀스 타입이 있다. 

# 2.1 깊은 복사와 슬라이싱 연산

# 2.1.1 가변성

# 튜플, 문자열, 바이트는 불변 객체 타입, 리스트와 바이트는 가변 객체 타입. 
# 불변 객체 타입은 가변 객체 타입보다 효율적임. 
# 일부 컬렉션 데이터 타입은 불변 데이터 타입으로 인덱싱할 수 있음. 

# ex. 깊은 복사
myList = [1,2,3,4]
newList = myList[:]
newList2 = list(newList)
print(myList)
print(newList)
print(newList2) # 모두 다 [1,2,3,4]로 나온다. 

people = {'버피','에인절','자일스'}
slayers = people.copy()
slayers.discard('자일스')
slayers.remove('에인절')
print(slayers) # {'버피}
print(people) # {'버피','에인절','자일스'}

# dict 깊은 복사
myDict = {'안녕':'세상'}
newDict = myDict.copy()
print(newDict) # {'안녕': '세상'}

# 기타 객체에 대한 깊은 복사는 copy모듈 사용
import copy
myObj = '다른 어떤 객체'
newObj = copy.copy(myObj) # Shallow copy
newObj2 = copy.deepcopy(myObj) # Deep copy
print(newObj) # 다른 어떤 객체
print(newObj2) # 다른 어떤 객체 

# 2.1.2 슬라이싱 연산자 

# ex
word = '뱀파이어를 조심해!'
print(word[-1]) # !
print(word[-2]) # 해
print(word[-2:]) # 해!
print(word[:-2]) # 뱀파이어를 조심
print(word[-0]) # 뱀

# 2.2 문자열 

# 문자열은 str로 표현한다. 
# string 형식은 사람을 위해 설계, representational 형식은 파이썬 인터프리터에서 사용하는 문자열로 보통 디버깅시 사용된다. 

# 2.2.1 유니코드 문자열

# 유니코드는 전 세계 언어의 문자를 정의하기 위한 국제 표준 코드. 문자열 앞에 u를 붙이면 유니코드 문자열을 만들 수 있다. 

# ex 
print(u'잘가\u0020세상 !') # 잘가 세상 !
# 이스케이프 시퀀스는 서수 값이 0x0020인 유니코드 문자를 나타낸다. 
# 일반적인 아스키 코드의 표현은 7비트, 유니코드는 16비트 

# 2.2.2 문자열 메서드

# ex) join 문 
slayers = ['버피','에인절','자일스']
print(" ".join(slayers)) # 버피 에인절 자일스
print('-<>-'.join(slayers)) # 버피-<>-에인절-<>-자일스
print(''.join(slayers)) # 버피에인절자일스

# join 메서드와 내장 함수 reversed() 메서드를 같이 사용 가능.
print(' '.join(reversed(slayers))) #자일스 에인절 버피

# ljust(), rjust()
# A.ljust(width, fillchar)는 문자열 A '맨 처음'부터 문자열을 포함한 길이 width 만큼 문자 fillchar를 채운다. 
# A.rjust(width, fillchar)는 문자열 A '맨 끝'부터 문자열을 포함한 길이 width 만큼 문자 fillchar를 채운다.

# ex

name = '스칼렛'
print(name.ljust(50,'-')) # 스칼렛-----------------------------------------------
print(name.rjust(50,'-')) # -----------------------------------------------스칼렛

# format()
# A.format()은 문자열 A에 변수를 추가하거나 형식화하는 데 사용된다. 

# ex
print("{0} {1}".format('안녕','파이썬')) # 안녕 파이썬
print('이름 : {who}, 나이 : {age}'.format(who='제임스', age = 17)) # 이름 : 제임스, 나이 : 17
print('이름 : {who}, 나이 : {0}'.format(12, who = '에이미')) # 이름 : 에이미, 나이 : 12
print('{} {} {}'.format('파이썬','자료구조','알고리즘')) # 파이썬 자료구조 알고리즘

# + 연사자를 사용하면 문자열을 좀 더 간결하게 결합할 수 있다. 
# format()은 3개의 지정자로 문자열을 조금 더 유연하게 결합할 수 있다. 
# s는 문자열 형식, r은 표현 형식, a는 아스키 코드 형식을 의미함. 

# ex
import decimal
print("{0} {0!s} {0!r} {0!a}".format(decimal.Decimal('99.9'))) # 99.9 99.9 Decimal('99.9') Decimal('99.9')

# 문자열 언패킹 
# 문자열 매핑 언패킹 연산자는 **이다. 사용하면 함수로 전달하기에 적합한 키-값 딕셔너리가 생성된다. 
# local 매서드는 현재 스코프에 있는 지역변수를 딕셔너리로 반환해준다. 

hero = '버피'
number = 999
print('{number} : {hero}'.format(**locals())) # 999 : 버피

# splitnes()
slayers = '로미오\n줄리엣'
print(slayers.splitlines()) # ['로미오', '줄리엣']

# split()
# A.split(t, n)은 문자열 A에서 문자열 t를 기준으로 정수 n번만큼 분리한 문자 열 리스트를 반환한다. 
# n을 지정해주지 않으면 대상 문자열을 t로 최대한 분리한다. 
# t도 지정하지 않으면 공백 문자로 구분한 문자열 리스트를 반환한다.

slayers = '버피*크리스-메리*16'
fields = slayers.split("*")
print(fields) # '버피', '크리스-메리', '16']
job = fields[1].split('-') 
print(job) # ['크리스', '메리']

# strip() 공백이면 공백 문자를 제거함. 
slayers = '로미오 & 줄리엣999'
print(slayers.strip('999')) # 로미오 & 줄리엣
# lstrip(), rstrip도 있다. 

# swapcase() 메서드 
# 대소문자를 반전 시켜주는 기능 
slayers = 'Buffy and Faith'
print(slayers.swapcase()) # bUFFY AND fAITH

# capitalize()는 첫 글자를 대문자로, lower()는 문장 전체를 소문자, upper()는 문장 전체를 대문자 

# index(), find()

# A.index(sub,start,end)는 문자열 A에서 부분 문자열 sub의 인덱스 위치를 반환하며, 실패하면 ValueError 예외를 발생시킨다. 
# A.find(sub, start, end)는 문자열 A에서 부분 문자열 sub의 인덱스 위치를 반환하며, 실패하면 -1을 반환한다. 
# start와 end는 문자열 범위이고, 생략할 경우 전체 문자열에서 부분 문자열 sub를 찾는다. 

# ex
slayers = 'Buffy and Faith'
print(slayers.find('y')) # 4
print(slayers.find('k')) # -1
print(slayers.index('k')) # ValueError
print(slayers.index('y')) # 4

# rindex()와 rfind()도 있다. 

# count() 메서드 
# A.count(sub, start, end)

slayers = 'Buffy is Buffy is Buffy'
print(slayers.count('Buffy',0,-1)) # 2
print(slayers.count('Buffy')) # 3

# replace(old, new, maxreplace) 메서드
# maxreplace 만큼 변경해준다. 없으면 전부다 변함
slayers = 'Buffy is Buffy is Buffy' 
print(slayers.replace('Buffy','who',2)) # who is who is Buffy

# f-strings
# .format 방식에 비해서 빠르고 간결하다. 

name = '프레드'
print(f'그의 이름은 {name!r}입니다.') # 그의 이름은 '프레드'입니다.
print(f'그의 이름은 {repr(name)}입니다.') # !r과 repr()은 같다.

import decimal
width = 10
precision = 4
value = decimal.Decimal('12.34567')
print(f'결과: {value:{width}.{precision}}') # 중첩 필드 사용, 
                                           # 결과:      12.35

from datetime import datetime
today = datetime(year = 2020, month = 2, day = 27)
print(f'{today : %B %d, %Y}') #날짜 포맷 지정 지정자 사용
                              # February 27, 2020
number = 1024
print(f'{number:#0x}') # 0x400 
                       # 정수 포맷 지정자 사용 (16진수 표현)
# 2.3 튜플 
# 쉼표로 구분된 값으로 이루어지는 불변 시퀀스 타입. 각 위치에 개첵 참조를 가지고 있다. 
# 리스트와 같이 변경 가능한 객체를 포함하는 튜플을 만들 수 있다. 

# 2.3.1 튜플 메서드
# A.count(x) 튜플 A에 담긴 x의 개수를 반환한다.

# ex
t = 1,2,3,4,5,5,6,6,6,7
print(t.count(5)) # 2

# index(x) 매서드는 항목 x의 인덱스 위치를 반환한다.

print(t.index(7)) # 9

# 2.3.2 튜플 언패킹

# 모든 반복 가능한 객체는 시퀀스 언패킹 연산자 *를 사용하여 언패킹 할 수 있다. 

# ex

x , *y = (1,2,3,4)
print(x) # 1
print(y) # [2,3,4]

*x , y = (1,2,3,4)
print(x) # [1,2,3]
print(y) # 4

# 2.3.3 네임드 튜플 

# collections에는 네임드 튜플이라는 시퀀스 데이터 타입이 있다. 
# 일반 튜플과 비슷한 성능을 가지고 있지만, 튜플 항목을 인덱스 위치뿐만 아니라 이름으로도 참조할 수 있다. 

# collections.namedtuple(): 첫번째 인수는 만들고자 하는 사용자 정의 튜플 데이터 타입의 이름. 
# 두번째 인수는 사용자 저의 튜플 각 항목을 지정하는 공백으로 구분된 문자열이다. 

# ex
import collections
person = collections.namedtuple('Person','name age gender')
p = person('아스틴', 30, '남자') # Person(name='아스틴', age=30, gender='남자')
print(p) 
print(p[0]) # 아스틴
print(p.name) # 아스틴
p.age = 20 # 에러가 발생하고, 일반 튜플과 마찬가지로 불변형이다.


# 2.4 리스트

# 리스트는 크기를 동적으로 조정할 수 있는 배열이다. 연결 리스트와는 아무런 관련 없음.
# 연결리스트는 매우 중요한 추상 데이터 타입(ADT)이다. 
# 리스트는 가변 타입이다. 

# 리스트에 항목을 추가하거나 제거할 때는 append()와 pop()을 사용한다. 시간복잡도는 O(1)이다. 
# 리스트 항목을 검사 해야되는 remove(), index(), in등은 시간복잡도가 O(n)이다.
# insert()는 지정한 인덱스에 항목을 삽입하고, 그 이후의 인덱스 항목들을 한 칸 씩 뒤로 밀어야하기 때문에 시간복잡도는 O(n)이다. 
# 검색이나 멤버십 테스트 시 빠른 속도가 필요하면 set이나 dict 같은 컬렉션 타입이 더 좋을 수 있다. 

# 2.4.1 리스트 매서드

# a.append(x)는 a[len(a):] = [x]와 같은 의미다. 리스트 끝에 값을 추가해준다. 

# ex
people = ['버피','자일스']
people.append('페이스')
print(people) # ['버피', '자일스', '페이스']
people[len(people):] = ['잰더']
print(people) # ['버피', '자일스', '페이스', '잰더']

# extend()는 반복 가능한 모든 항목을 리스트에 추가해주는 기능이다. 
# a.extend(x) 는 a[len(a):] = c 또는 a += c와 같다. 

# ex

people = ['버피','자일스']
people.extend('페이스')
print(people) # ['버피', '자일스', '페', '이', '스']
people += '월로'
print(people) # ['버피', '자일스', '페', '이', '스', '월', '로']
people += ['젠더']
print(people) # ['버피', '자일스', '페', '이', '스', '월', '로', '젠더']
people[len(people):] = '아스틴'
print(people) # ['버피', '자일스', '페', '이', '스', '월', '로', '젠더', '아', '스', '틴']

# insert(i, x)는 리스트에 인덱스 위치 i에 x를 삽입한다. 

people = ['버피','자일스']
people.insert(1, '페이스')
print(people) # ['버피', '페이스', '자일스']

# remove()

people = ['버피','자일스']
people.remove('자일스')
print(people) # ['버피']

# pop(x) 인덱스 x에 있는 항목을 제거하고 그 값을 반환. 따로 표시 안해주면 맨 끝 항목을 제거한다. 

people = ['버피','자일스']
people.pop('자일스')
print(people) # ['버피']

# index() 인덱스를 반환해준다. 

# count(x) x가 얼마나 있는지 세준다. 

# sort()는 오른차순으로 정렬해준다. sort(reverse=True)면 내림차순으로 정렬. 

# reverse()는 리스트의 항목을 반전시켜준다. list[::1]과 같다. 


# 2.4.2 리스트 언패킹

# 튜플 언패킹과 비슷. 

# 2.4.3 리스트 컴프리핸션

a = [y for y in range(1900, 1940) if y%4 == 0]
print(a) # [1900, 1904, 1908, 1912, 1916, 1920, 1924, 1928, 1932, 1936]
b = [2**i for i in range(13)] # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
print(b)
c = 'words is awesome and a vammpire slayer'.split()
e = [[y.upper(),y.lower(), len(y)] for y in c]
for i in e:
    print(i)

# 2.5 바이트와 바이트 배열 

# 불변 타입의 바이트와 가변 타입의 바이트 배열을 제공함. 
# 두 타입 모두 0~255 범위의 부호 없는 8비트 정수 시퀀스로 이루어짐. 
# 바이트 타입은 문자열 타입과 유사하고, 바이트 배열 타입은 리스트 타입과 유사함. 

# 2.5.1 비트와 비트 연산자 

# 비트 연산자는 비트로 표현된 숫자를 조작하는데 유용함.

# 2.6 연습문제 

def revert(s):
    if s:
        s = s[-1] + revert(s[:-1])
    return s

def revert2(string):
    return string[::-1]

if __name__ == '__main__':
    str1 = '안녕 세상!'
    str2 = revert(str1)
    str3 = revert2(str1)
    print(str2)
    print(str3)

s = '파이썬 알고리즘 정말 재미있다!'

def split_sentence(s):
    s = s.split()
    return s

def reverse_sentence(s):
    s.reverse()
    return ' '.join(s)

s = split_sentence(s)
s = reverse_sentence(s)
print(s)

# 단순 문자열 압축

a = 'aabcccccaaa'

def str_compression(s):
    count, last = 1, ''
    list_aux = []
    for i, c in enumerate(s):
        if last == c:
            count += 1
        else:
            if i != 0:
                list_aux.append(str(count))
            list_aux.append(c)
            count = 1
            last = c
    list_aux.append(str(count))
    return ''.join(list_aux)

print(str_compression(a))

# 문자열 순열 

import itertools

def perm(s):
    if len(s) < 2:
        return s 
    res = []
    for i, c in enumerate(s):
        for cc in perm(s[:i] + s[i+1:]):
            res.append(c+cc)
    return res

def perm2(s):
    res = itertools.permutations(s)
    return ["".join(i) for i in res]

s = '012'
print(perm(s))
print(perm2(s))


def combinations(s):
    if len(s) < 2:
        return s
    res = []
    for i,c in enumerate(s):
        res.append(c)
        for j in combinations(s[:i] + s[i+1:]):
            res.append(c + j)
    return res

print(combinations('abc'))

# 2.6.5 회문

def is_palindrome(s):
    l = s.split(' ')
    s2 = ''.join(l)
    return s2 == s2[::-1] 

def is_palindrome2(s):
    l = len(s)
    f, b = 0, l-1
    while f < l // 2:
        while s[f] == ' ':
            f += 1
        while s[b] == ' ':
            b -= 1
        if s[f] != s[b]:
            return False
        f+=1
        b-=1
    return True

def is_palindrome3(s):
    s = s.strip()
    if len(s) < 2:
        return True
    if s[0] == s[-1]:
        return is_palindrome(s[1:-1])
    else:
        return False 
    