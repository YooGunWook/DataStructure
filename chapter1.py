# 1.1 정수

# 파이썬에서 정수는 immutable type. 
# 바이트 수 체크할 때는 (정수).bit_length로 확인한다. 

# ex
print((999).bit_length())
# 결과는 10이 나온다. 10 바이트가 필요한거임. 

# 문자열을 숫자열로 변환하거나, 다른 진법의 문자열을 정수(10진법)으로 변환하려면 int(문자열, 밑).

# ex 
s = '11'
d = int(s)
print(d) # 11이 출력된다. 

b = int(s, 2)
print(b) # 3이 출력된다. 밑은 2에서 36 사이의 선택적 인수. 
# 밑 범위의 숫자를 벗어나는 값을 입력하면 int에서 ValueError 발생한다. 

# 1.2 부동소수점

# 파이썬에서 부동소수점은 float으로 나타낸다. 불변형임. 
# 32비트 부동소수점을 나타낼 때, 1비트는 부호(1:양수, 0:음수) 23비트는 유효 숫자 자리수 8비트는 지수다. 

# 1.2.1 부동소수점끼리 비교하기

# 이진수 분수로 표현되기 때문에 함부로 비교하거나 빼면 안됨. 
# ex
print(0.2 * 3 == 0.6) # False
print(1.2 - 0.2 == 1.0) # True
print(1.2 - 0.1 == 1.1) # False
print(0.1 * 0.1 == 0.01) # True

# 1.2.2 정수와 부동소수점 메서드 

# divmod(x,y)를 사용하면 몫과 나머지를 반환한다. 
print(divmod(45,6)) # (7,3)

# round(x, n) n이 음수면 x를 n만큼 반올림한 값 반환. n이 양수면 x를 소수점 이하 n자리로 반올림한 값 변환. 
print(round(100.96, -2)) # 100.0
print(round(100.96, 2)) # 100.96

# 1.3 복소수 

# 파이썬 복소수는 부동소수점 한 쌍을 갖는 불변형임. z.real, z.imag, z.conjugate() 같은 메서드로 실수부, 허수부, 켤레 복소수를 구할 수 있다. 
# 복소수 사용할 때 cmath 모듈을 사용해야함. math에 있는 대부분의 삼각함수와 로그함수의 복소수 버전을 제공한다. 
# cmath.phase(), cmath.polar(), cmath.rect(), cmath.pi, cmath.e와 같은 복소수 전용 함수를 제공한다. 

# 1.4 fraction 모듈 

# 분수를 다룰 때 사용함. 
# ex 
from fractions import Fraction

def rounding_floats(number1, places):
    return round(number1, places)

def float_to_fractions(number):
    return Fraction(*number.as_integer_ratio())

def get_denominator(number1, number2):
    a = Fraction(number1, number2)
    return a.denominator # -> 분모를 반환한다. 

def get_numerator(number1, number2):
    a = Fraction(number1, number2)
    return a.numerator # -> 분자를 반환한다.

def test_testing_floats():
    number1 = 1.25
    number2 = 1
    number3 = -1
    number4 = 5/4
    number6 = 6
    assert(rounding_floats(number1,number2) == 1.2)
    assert(rounding_floats(number1*10, number3) == 10)
    assert(float_to_fractions(number1)==number4)
    assert(get_denominator(number2,number6) == number6)
    assert(get_numerator(number2, number6) == number2)
    print('테스트 통과!')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_testing_floats()

# 1.5 decimal 모듈 

# 정확한 10진법의 부동소수점 숫자가 필요한 경우 decimal.Decimal을 사용한다. 
# 정수 또는 문자열을 인수로 취한다. 
# 부동소수점의 반올림, 비교, 뺄셈 등에서 나타나는 문제를 효율적으로 처리할 수 있음. 

# ex
print(sum(0.1 for i in range(10)) == 1.0) # False
from decimal import Decimal
print(sum(Decimal("0.1") for i in range(10)) == Decimal('1.0')) # True

# 1.6 2진수, 8진수, 16진수

# bin(i)는 정수 i의 2진수 문자열을 반환한다. 

# ex 
print(bin(999)) # 0b1111100111
print(oct(999)) # 0o1747

# 1.7 연습문제

# 10 진수로 변환 
def convert_to_decimal(number, base):
    multiplier, result = 1, 0 
    while number > 0:
        result += number % 10 * multiplier # -> 가장 핵심 부분. 반복문을 돌면서 
        # number % 10(base)로 일의 자리 숫자를 하나씩 가져와서 계산함. 
        # 반복문 계산을 위해 number - number // 10을 사용한다. 
        multiplier *= base
        number = number // 10
    return result

def test_convert_decimal():
    number, base = 1001, 2 
    assert(convert_to_decimal(number, base) == 9)
    print('테스트 통과')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_convert_decimal()

# 10진수를 다른 진법의 숫자로 변환 

def convert_from_decimal(number, base):
    multiplier, result = 1, 0
    while number > 0 :
        result += number % base * multiplier # base가 10 이상인 경우 숫자가 아니라 문자로 나타내야한다. 
        multiplier *= 10
        number = number // base
    return result

def test_convert_from_decimal():
    number, base = 9, 2 
    assert(convert_from_decimal(number, base) == 1001)
    print('테스트 통과')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_convert_from_decimal()

# 10진법 숫자를 20 이하의 진법으로 변환

def convert_from_decimal_larger_bases(number,base):
    strings = '0123456789ABCDEFGHIJ'
    result = ""
    while number > 0:
        digit = number % base
        result = strings[digit] + result
        number = number // base
    return result

def test_convert_from_decimal_larger_bases():
    number, base = 31, 16
    assert(convert_from_decimal_larger_bases(number, base) == '1F' )
    print('테스트 통과')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_convert_from_decimal_larger_bases()

# 재귀 함수를 사용한 진법 변환

def convert_dec_to_any_base_rec(number, base):
    convertString = '0123456789ABCDEF'
    if number < base :
        return convertString[number]
    else:
        return convert_dec_to_any_base_rec(number // base, base) + convertString[number % base]

def test_convert_dec_to_any_base_rec():
    number = 9
    base = 2
    assert(convert_dec_to_any_base_rec(number, base) == '1001')
    print('테스트 통과')


if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_convert_dec_to_any_base_rec()

# 1.7.2 최대공약수 

def finding_gcd(a,b):
    while(b != 0):
        result = b
        a,b = b, a % b
    return result

def test_finding_gcd():
    number1 = 21
    number2 = 12
    assert(finding_gcd(number1,number2) == 3)
    print('테스트 통과')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_finding_gcd()

# 1.7.3 random 모듈 

import random

def testing_random():
    # random test
    values = [1,2,3,4]
    print(random.choices(values))
    print(random.choices(values))
    print(random.choices(values))
    print(random.sample(values,2))
    print(random.sample(values,3))

    # value shuffling
    random.shuffle(values)
    print(values)

    # 임의의 정수 추출
    print(random.randint(0,10))
    print(random.randint(0,10))

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    testing_random()


# 1.7.4 피보나치 수열

# 피보나치 수열은 첫째 및 둘째 항이 1이며, 그 이후의 모든 항은 바로 앞 두 항의 합인 수열. 

# ex
import math

def find_fibonacci_seq_iter(n): # 반복문을 사용하고, 함수의 시간 복잡도는 O(n)
    if n < 2 :
        return n
    a,b = 0,1
    for i in range(n):
        a,b = b,a+b
    return a

def find_fibonacci_seq_rec(n): # 재귀 함수 호출을 사용하고, 함수의 시간 복잡도는 O(2^n)
    if n < 2: 
        return n
    return find_fibonacci_seq_rec(n-1) + find_fibonacci_seq_rec(n-2)

def find_fibonacci_seq_form(n): # 수식을 사용하고 함수의 시간 복잡도는 O(1)이다. 그러나 70 이후의 결과는 정확하지 않다. 
    sq5 = math.sqrt(5)
    phi = (1+sq5) / 2
    return int(math.floor(phi**n/sq5))

def test_find_fib():
    n = 10
    assert(find_fibonacci_seq_rec(n)==55)
    assert(find_fibonacci_seq_iter(n)==55)
    assert(find_fibonacci_seq_form(n)==55)
    print('테스트 통과')

if __name__ == "__main__": # --> 이 코드는 현재 스크립트 파일이 실행되는 상태를 파악하기 위해 사용함.
    test_find_fib()

# generator를 통해서도 피보나치 수열을 구할 수 있다.
# generator는 파이썬의 시퀀스를 생성하는 객체다. 전체 시퀀스를 한번에 메모리에 생성하고 정렬할 필요 없이, 잠재적으로 아주 큰 시퀀스를 순회할 수 있음. 
# 제너레이터 함수는 yield

# ex

def fib_generator():
    a,b = 0,1
    while True:
        yield b
        a,b = b, a+b

if __name__ == "__main":
    fg = fib_generator()
    for _ in range(10):
        print(next(fg), end = " ")


# 1.7.5 소수

# 소수는 자신보다 작은 두 개의 자연수를 곱하여 만들 수 없는 1보다 큰 자연수다. 

# ex 페르마 소정리

import math
import random

def finding_prime(number): # 일반적으로 소수를 찾는 법. 4 미만이면 무조건 소수기 때문에 True 반환, 그 외는 나머지가 0이면 False 소수가 아닌거고, 그 외는 소수를 반환하게 하는 것이다. 
    num = abs(number)
    if num < 4 :
        return True
    for x in range(2 , num):
        if num % x == 0:
            return False
        return True

def finding_prime_sqrt(number): # sqrt를 이용해서 소수를 찾는 법. 
    num = abs(number)
    if num < 4: 
        return True
    for x in range(2 , int(math.sqrt(num))+1):
        if number % x == 0:
            return False
        return True

def finding_prime_fermat(number):
    if number <= 102:
        for a in range(2, number):
            if pow(a, number-1, number) != 1:
                return False
        return True
    else:
        for i in range(100):
            a = random.randint(2, number - 1)
            if pow(a, number-1, number) != 1:
                return False
            return True

def test_finding_prime():
    number1 = 17 
    number2 = 20 
    assert(finding_prime(number1) is True)
    assert(finding_prime(number2) is False)
    assert(finding_prime_sqrt(number1) is True)
    assert(finding_prime_sqrt(number2) is False)
    assert(finding_prime_fermat(number1) is True)
    assert(finding_prime_fermat(number2) is False)
    print('테스트 통과')

if __name__ == "__main__":
    test_finding_prime()

import math
import random
import sys

def finding_prime_sqrt(number): # sqrt를 이용해서 소수를 찾는 법. 
    num = abs(number)
    if num < 4: 
        return True
    for x in range(2 , int(math.sqrt(num))+1):
        if number % x == 0:
            return False
        return True

def generate_prime(number = 3):
    while 1:
        p = random.randint(pow(2,number -2), pow(2, number-1)-1)
        p = 2 * p + 1
        if finding_prime_sqrt(p):
            return p

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: generate_prime.py number")
        sys.exit()
    else:
        number = int(sys.argv[1])
        print(generate_prime(number))
