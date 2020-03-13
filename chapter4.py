# 4.1 모듈

# 파이썬에서 모듈은 def를 사용해서 정의한다. 실행되면 함수의 객체와 참조가 같이 생성된다. 
# 반환값을 정의하지 않으면, 자동으로 None으로 반환한다. 이를 프로시져라고 부른다. 

# 4.1.1 스택과 활성화 레코드 

# 함수가 호출될 때마다 활성화 레코드가 생성된다. 
# 활성화 레코드에는 함수의 정보가 기록되며, 이를 스택에 저장한다. 

# 다음과 같은 순서로 처리된다. 
# 1) 함수의 실제 매개변수를 스택에 저장한다.
# 2) 반환 주소를 스택에 저장한다. 
# 3) 스택의 최상위 인덱스를 함수의 지역 변수에 필요한 총량만큼 늘린다.
# 4) 함수로 건너뛴다.

# 활성화 레코드를 풀어내는 절차는 다음과 같다. 
# 1) 스택의 최상위 인덱스는 함수에 소비된 총 메모리량(지역 변수)만큼 감소한다. 
# 2) 반환 주소를 스택에서 빼낸다. 
# 3) 스택의 최상위 인덱스는 함수의 실제 매개변수만큼 감소한다. 

# 4.1.2 모듈의 기본값

# 모듈을 생성할 떼, 함수 또는 메서드에서 가변 객체를 기본값으로 사용해선 안된다. 

# 나쁜예

def append(number,number_list=[]):
    number_list.append(number)
    return number_list

print(append(5)) # [5] 예상결과 [5]
print(append(7)) # [5, 7] 예상결과 [7]
print(append(2)) # [5, 7, 2] 예상결과 [2]

# 좋은예
def append(number,number_list=None):
    if number_list is None:
        number_list = []
    number_list.append(number)
    return number_list

print(append(5)) # [5] 
print(append(7)) # [7]
print(append(2)) # [2]

# 4.1.3 __init__.py 파일

# 패키지는 모듈과 __init__.py 파일이 있는 디렉터리다. 
# 파이썬은 이 파일이 있는 디렉터리를 패키지로 취급한다.
# 모듈 검색 경로 중 string과 같이 흔한 이름의 디렉터리에 유효한 모듈이 있는 경우 이러한 모둘이 검색되지 않는 문제를 방지하기 위해서다.
# import 폴더이름.파일모듈명

# __init__.py 파일은 빈 파일일 수도 있지만, 패키지의 초기화 코드를 실행하거나, __all__변수를 정의할 수도 있다. 
#__all__ = ['파일1',.....]

# 실제 파일 이름은 확장자가 .py지만 여기서 작성활 때는 .py를 붙이지 않는다. 
# from 폴더이름 import *
# 이름이 __로 시작하는 모듈을 제외한 모듈의 모든 객체를 불러온다. __all__ 변수가 있는 경우, 해당 리스트의 객체를 불러온다. 

# 4.1.4 __name__ 변수
# 파이썬은 모듈을 임포트할 때 마다 __name__이라는 변수를 만들고, 모듈 이름을 저장한다. 
# 대화식 인터프리터 또는 .py 파일을 직접 실행하면 파이썬은 __name__을 __main__으로 설정한다. 따라서 조건문에서 참에 해당하는 코드를 실행한다. 

# 4.1.5 컴파일된 바이트코드 모듈
# 바이트 컴파일 코드는 표준 모듈을 많이 사용하는 프로글매의 시작 시간을 줄이기 위한 것임. 
# -0 플래그를 사용하여 파이썬 인터프리터를 호출하면, 최적화된 코드가 생성되어 .pyo파일이 저장된다. 3.5 버전 이후부터는 .pyc를 사용한다. 


# 4.1.6 sys 모듈
# sys.path는 인터프리터가 모듈을 검색할 경로를 담은 문자열 리스트다. 
# 이 변수를 사용하면 PYTHONPATH 환경변수 또는 내장된 기본값 경로로 초기화된다. 
# 환경변수를 수정하면 모듈 경로를 추가하거나 임시로 모듈 경로를 추가할 수 있다. 
# sys.ps1과 sys.ps2는 파이썬 대화식 인터프리터의 기본 및 보조 프롬프트 문자열을 정의한다. 기본값은 각각 >>>및 ...이다. 

# sys.argv 함수를 사용하면 명령 줄에 전달된 인수를 프로그램 내에세도 사용할 수 있다. 
import sys
def main():
    for arg in sys.argv[1:]:
        print(arg)

if __name__ == '__main__':
    main()

# dir() 내장 함수는 모듈이 정의하는 모든 유형의 이름(모듈,변수,함수)을 찾는데 사용된다. 이름 기준으로 정렬된 문자열 리스트를 반환한다. 
# dir() 함수는 내장 함수 및 변수의 이름까지는 나열하지 않는다. 객체의 모든 메서드나 속성을 찾는 데 유용하다. 

# 4.2 제어문

# 4.2.1 if문
# 다른 언어의 switch나 case문을 대체한다.

# 4.2.2 for문
# 파이썬의 for문은 모든 시퀀스 항목을 순서대로 순회한다. 

# 4.2.3 참과 거짓
# False는 숫자 0 또는 특수객체 None, 빈 컬렉션 시퀀스에 의해 정의된다. 여기에 속하지 않으면 True다. 
# == 또는 != 연산자를 사용하여 내장변수 None 같은 싱글턴을 비교하지 않는다. 대신 is나 is not을 쓴다. 
# if x is not None과 if x를 잘 구분해서 사용한다. 
# ==을 사용하여 불리언 변수를 Fasle와 비교하지 않는다. 대신 if not x 를 사용한다. 
# None과 False를 구별할 필요가 있는 경우 if not x and x is not None과 같은 연결 표현식을 사용한다. 
# 시퀀스의 경우, 빈 시퀀스는 False다. 
# if len(시퀀스) if not len(시퀀스) 보다는 if 시퀀스 또는 if not 시퀀스로 처리해준다. 
# 정수 처리 시 뜻하지 않게 None을 0으로 잘못 처리하는 것처럼 암묵적 False를 사용하는 것은 위험하다. 

## 좋은 예
if not users:
    print('사용자가 없습니다.')
if foo == 0:
    hadle_zero()

if i % 10 == 0:
    handle_multiple_of_ten()

## 나쁜 예
if len(users) == 0:
    print('사용자가 없습니다.')

if foo is not None and not foo:
    hadle_zero()

if not i % 10:
    handle_multiple_of_ten()


# 4.2.4 return과 yield
# 파이썬에서 제너레이터는 이터레이터를 작성하는 편리한 방법이다. 
# return 키워드는 반환값을 반환하고 메서드를 종료한 후 호출자에게 제어를 반환한다. 
# yield 키워드는 각 반환값을 호출자에게 반환하고, 반환값이 모두 소진되었을 때에만 메서드가 종료된다. 
# 이터레이터는 프로토콜을 구현하는 컨테이너 객체라고 할 수 있는데, 컨테이너의 다음 값을 반환하는 __next()__() 메서드와 이터레이터 자신을 반환하는 __iter__()메서드를 기반으로 한다. 
# 제너레이터는 최종값을 반환하지만, 이터레이터는 yield 키워드를 사용해서 코드 실행 중에 값을 반환한다. 
# __next__() 메서드를 호출할 때마다 어떤 값 하나를 추출한 후 해당 yield 표현식의 값을 반환한다. 이터레이터는 예외가 발생할 때까지 발생한다. 

a = [1,2,3]
def f(a):
    while a:
        yield a.pop()

# 시퀀스를 반환하거나 반복문을 사용하는 함수를 다룰 때 제너레이터를 고려할 수 있다. 

def fib_generator():
    a,b = 0,1
    while True:
        yield b
        a, b = b, a+b
if __name__ == '__main__':
    fib = fib_generator()
    print(next(fib)) # 1
    print(next(fib)) # 1
    print(next(fib)) # 2
    print(next(fib)) # 3

# 4.2.5 break 대 continue

# 반복문에서 break 키워드를 만나게 되면 바로 반복문을 빠져 나가게 된다. 
# continue 키워드를 만나게 되면 반복문의 다음 단계로 전환하게 된다. 
# else는 반복문이 종료되었을 때 (for문에서는 리스트의 항목을 모두 순회 했을 때, while문에서는 조건이 false일 때) 발생된다. 
# break로 종료될 경우 실행되지 않는다.

# 4.2.6 range()
# 숫자 리스트를 생성할 때 쓰는 함수. 숫자 시퀀스를 순회할 때 유용하다.

# 4.2.7 enumerate()
# 반복 가능한 객체의 인덱스 값과 항목 값의 튜플을 반환한다. 

# 4.2.8 zip()
# 2개 이상의 시퀀스를 인수로 취하고, 짧은 길이의 시퀀스를 기준으로 각 항목이 순서대로 1:1 대응하는 새로운 튜플 시퀀스를 만든다. 

a = [1,2,3,4,5]
b = ['a','b','c','d','e']

print(list(zip(a,b))) # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd'), (5, 'e')]

# 4.2.9 filter()
# 시퀀스 항목들 중 함수 조건이 참인 항목만 추출해서 구성된 시퀀스를 반환한다. 

def f(x):
    return x % 2 != 0 and x % 3 != 0

print(f(33)) # False
print(f(17)) # True
print(list(filter(f,range(2,25)))) # [5, 7, 11, 13, 17, 19, 23]

# 4.2.10 map()

# map(function, list)는 시퀀스의 모든 항목에 함수를 적용한 결과 리스트를 반환한다.

def cube(x): 
    return x*x*x

print(list(map(cube,range(1,11)))) # [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]

seq = range(8)
def square(x):
    return x*x
print(list(zip(seq,map(square,seq)))) # [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25), (6, 36), (7, 49)]

# 4.2.11 람다 함수
# 람다 함수를 쓰면 코드 내에서 함수를 간결하게 동적으로 사용할 수 있다. 

area = lambda b, h: 0.5 * b * h
print(area(5,4)) # 10.0

# defaultdict에서 키 생성시 매우 유용하다(누락된 키에 대한 기본 값 설정 시). 

import collections
minus_one_dict = collections.defaultdict(lambda : -1)
point_zero_dict = collections.defaultdict(lambda : (0,0))
message_dict = collections.defaultdict(lambda : 'No message')

# 4.3.1 파일 처리 메서드

# open(filename, mode, encoding)
# 파일 객체를 반환하는 메서드다. 모드와 인코딩은 옵션이며, 생략하면 텍스트 읽기 모드와 시스템 기본 형식 인코딩이 적용된다. 
# r : 읽기 모드, w : 쓰기 모드, a : 추가 모드, r+ : 읽기와 쓰기 모드 , t : 텍스트 모드, b : 바이너리 모드

# read(size)는 size만큼 내용을 읽고, 문자열로 반환한다. (파이썬3에서는 텍스트 모드의 경우 문자열로 반환하고, 바이너리 모드의 경우 바이트 객체를 반환한다.)
# size는 선택적 인수다. 생략하면 전체 내용을 읽고 반환한다. 

# readline()은 파일에서 한줄을 읽는다. 개행 문자는 문자열의 끝에 남으며, 파일의 마지막 행에서만 생략된다. 반환값이 모호해지는 문제가 있다. 

# readlines()는 파일의 모든 데이터 행을 포함한 리스트를 반환한다. size를 지정해주면 해당 바이트 수만큼 일고, 한 행을 완성하는데 필요한 만큼 더 읽어서 반환한다. 
# 메모리에 전체 파일을 불러올 필요 없이 줄 단위로 효율적으로 읽을 수 있ㄷ고, 완전한 행을 반환한다.

# write() 데이터를 파일에 쓰고 None을 반환한다. 바이너리 모드에서는 바이트 또는 바이트 배열 객체를 쓰고, 텍스트 모드에서는 문자열 객체를 쓴다. 

# tell(), seek()
# tell()은 파일의 현재 위치를 나타내는 정수를 반환한다. 파일의 위치는 시작부분에서 바이트 단위로 측정된다. 
# seek(offset, from-what) 메서드는 파일 내 탐색 위치를 변경할 때 사용한다. 
# 파일 위치는 기준이 되는 from-what애 offset을 더한 값으로 계산된다. 
# from-what의 경우 0은 기준이 파일의 처음 위치가 되고, 1은 파일의 현재위치, 2는 파일의 마지막 위치를 기준으로 삼게 된다.

# close()
# 파일을 닫고, 열린 파일이 차지하는 시스템 자원을 해체한다. 성공적으로 닫게 되면 True를 반환한다. 

# input()
# 사용자의 입력을 받는 함수다. 

# peek()
# peek(n)는 파일 포잍너 위치를 이동하지 않고, n바이트를 반환한다. 

# 4.3.2 shutil 모듈

# shutil 모듈은 시스템에서 파일을 조작할 때 유용하다. 

# 4.3.3 pickle 모듈
# pickle 모듈은 퍼아썬 객체를 가져와서 문자열 표현으로 반환한다. 
# 이 과정을 피클링이라고 한다. 
# 문자열 표현을 객체로 재구성하는 것을 언피클링이라고 한다. 

# 4.3.4 struct 모듈
# 파이썬 객체를 이진 표현으로 변환하거나, 이진 표현을 파이썬 객체로 변환할 수 있다. 객체는 특정 길이의 문자열만 처리할 수 있음.
# struct.pack() 함수는 struct 형식의 문자열과 값을 취하여 바이트 객체를 반환한다. 
# struct.unpack() 함수는 struct 형식의 문자열과 바이트 또는 바이트 배열 객체를 취하여 값을 반환한다. 
# struct.calcsize() struct 형식의 문자열을 취하여, struct 형식이 차지할 바이트 수를 반환한다. 

# 4.4 문자 처리
# 파이썬에는 두가지 오류가 있는데, 구문오류와 예외가 있다.

# 4.4.1 예외 처리
# 예외가 발생했는데 이를 코드 내에서 처리하지 않으면, 파이싼은 예외의 오류 메시지와 함꼐 트레이스백을 출력한다. 
# 파이썬에서는 try-except-finally로 예측 가능한 예외를 처리할 수 있다. 
# raise문을 사용해서 특정 예외를 의도적으로 발생시킬 수 있다, 