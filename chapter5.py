# 5. 객체지향 설계
# 데이터를 패키지화하고 메서드를 제한하는 것이 객체지향 프로그래밍. 

# 5.1 클래스와 객체
# 클래스는 사전에 정의된 특별한 데이터와 메서드의 집합이다. 
# 클래스에 선언된 모양 그대로 생성된 실체를 객체. 
# 객체가 스프트웨어에 실체화될 때 이 실체를 인스턴스라고 부름. 

# 가장 간단한 클래스 형태
class ClassName:
    # 문장 1
    # ...
    # 문장 n
    pass

x = ClassName()
print(x) # <__main__.ClassName object at 0x1083b4310>

# 5.1.1 클라스 인스턴스 생성

# 클래스 인스턴스 생성은 함수 표기법을 사용하여 초기 상태의 객체를 생성하는 일.
# 인스턴스 생성 작업은 어떤 특징을 가진 빈 객체를 만드는 것. 
# 여러 이름을 같은 객체에 바인딩 할 수 있음. 

# 속성
# 객체에는 데이터와 메서드로 이루어지는 클래스 속성이 있다. 
# 매서드 속성은 함수이고, 첫 번째 인수는 호출된 인스턴스 자신이다. 
# 속성은 점(.) 뒤에 나오는 모든 이름이다.
# 모듈명, 함수명과 같은 표현식에서 모듈명은 모듈 객체이고, 함수명은 객체의 속성 중 하나다. 

# 네임스페이스 
# 네임스페이스는 이름을 객체로 매핑하는 것임. 
# 대부분 네임스페이스는 파이썬 딕셔너리로 구현되어 있음. 
# 스크립트 파일이나 대화식 인터프리터의 최상위 호출에 의해 실행되는 명령문은 __main__이라는 모듈의 일부로 간주되어, 고유의 전역 네임스페이스를 갖는다. 

# 스코프
# 스코프는 네임스페이스에 직접 접근할 수 있는 파이썬 프로그램의 텍스트 영역이다. 
# 스코프는 정적으로 결정되지만, 동적을 사용된다. 
# 스코프는 텍스트에 따라 결정된다. 
# 한 모듈에 정의된 함수의 전역 스코프는 해당 모듈의 네임스페이스다. 
# 클래스 정의가 실행되면, 새로운 네임스페이스가 만들어지고, 지역 스코프로 사용된다. 

# 5.2 객체지향 프로그래밍

# 5.2.1 특수화 
# 특수화는 슈퍼 클래스 (부모 또는 베이스 클래스라고 한다.)의 모든 속성을 상속하여 새 클래스를 만드는 절차. 
# 모든 메서드는 서브 클래스에서 재정의 될 수 있다. 
# 구글 파이썬 스타일 가이드에서 한 클래스가 다른 클래스를 상속받지 않으면, 파이썬의 최상위 클래스인 object를 명시적으로 표기하는 것을 권장함. 

# 좋은 예
class SampleClass(object):
    pass

class OuterClass(object):
    class InnerClass(object):
        pass

class ChildClass(object):
    # 부모 클래스 상속

# 나쁜 예

class SampleClass():
    pass
class OuterClass:
    class InnerClass:
        pass

# 5.2.2 다형성

# 다형성(동적 메서드 바인딩)은 메서드가 서브 클래스 내에서 재정의 될 수 있다는 원리임. 
# 서브 클래스 객체에서 슈퍼 클래스와 동명의 메서드를 호출하면, 파이썬은 서브 클래스에 정의된 메서드를 사용한다는 뜻. 
# 슈퍼 클래스의 매서드를 호출하려면 내장된 super() 메서드를 통해 쉽게 호출 가능

class Symbol(object):
    def __init__(self,value):
        self.value = value

if __name__ == '__main__':
    x = Symbol('Py')
    y = Symbol('Py')

    symbols = set()
    symbols.add(x)
    symbols.add(y)

    print(x is y) # False
    print(x == y) # False
    print(len(symbols)) # 2

# x,y의 참조가 다르기 때문에 첫번째는 False가 나옴
# 두번째의 경우 값은 같지만 False가 나왔고, 세번째의 경우 중복 항목이 없기 때문에 1이 나와야할 거 같지만 2가 나왔다.

class Symbol(object):
    def __init__(self,value):
        self.value = value

    #__eq__는 객체의 비교를 담당한다. 
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return self.value == other.value
        else:
            return NotImplemented

if __name__ == '__main__':
    x = Symbol('Py')
    y = Symbol('Py')

    symbols = set()
    symbols.add(x)
    symbols.add(y)

    print(x is y) 
    print(x == y) 
    print(len(symbols)) 

# __eq__()매서드를 재정의하자 Symbol의 클래스가 해시 가능하지 않다고 에러가 발생한다. 
# 객체가 해시 가능하지 않다는 것은 가변 객체임을 의미하는데, 셋은 불변 객체다. 
class Symbol(object):
    def __init__(self,value):
        self.value = value

    #__eq__는 객체의 비교를 담당한다. 
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            return self.value == other.value
        else:
            return NotImplemented
    
    def __hash__(self):
        return hash(self.value)

if __name__ == '__main__':
    x = Symbol('Py')
    y = Symbol('Py')

    symbols = set()
    symbols.add(x)
    symbols.add(y)

    print(x is y) # False
    print(x == y) # True
    print(len(symbols)) # 1

# hash() 메서드를 넣었더니 원하는 결과가 출력됨. 

# 5.2.3 합성과 집합화
# 합성(집합화)는 한 클래스에서 다른 클래스의 인스턴스 변수를 포함하는 것을 말하며, 클래스 간의 관계를 나타낸다. 
# 파이썬의 모든 클래스는 상속을 사용한다. (object 베이스 클래스로부터 상속받는다.) 
# 대부분 클래스는 다양한 타입의 인스턴스 변수를 가지고, 합성과 집합화를 사용한다. 
# 두 클래스 A와 B가 있을때, 합성은 A와 B가 강한 연관 관계를 맺으며, 강한 생명주기를 가진다. 의존성이 강하다는 것이다. 
# 집합화는 A와 B가 연관성은 있지만 생명주기가 약하고 독립적이다. 

# 5.2.4 클래스 예제

import math

class Point(object):
    def __init__(self, x = 0, y = 0):
        self.x = x # 데이터 속성
        self.y = y

    def distance_from_origin(self): # 메서드 속성
        return math.hypot(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self): # repr 함수는 어떤 객체의 ‘출력될 수 있는 표현’(printable representation)을 문자열의 형태로 반환한다
        return "point ({0.x!r},{0.y!r}".format(self)

    def __str__(self):
        return "({0.x!r},{0.y!r}".format(self)

class CirCle(Point):
    def __init__(self,radius,x = 0, y = 0):
        super().__init__(x,y) # 생성 및 초기화
        self.radius = radius 

    def edge_distance_from_origin(self):
        return abs(self.distance_from_origin()-self.radius)

    def area(self):
        return math.pi*self.radius

    def circumference(self):
        return 2*math.pi*self.radius

    def __eq__(self,other):
        return self.radius == other.radius and super().__eq__(other)

    def repr(self):
        return 'circle ({0.radius!r}, {0.x!r}').format(self)

    def __str__(self):
        return repr(self)

# 5.3 디자인 패턴
# 디자인 패턴은 잘 설계된 구조의 형식적 정의를 소프트웨어 엔지니어링으로 옮긴 것이다. 

# 5.3.1 데커리이터 패턴
# 데커레이터 패턴은 @ 표기를 사용해 함수 또는 메서드의 변환을 우아하게 지정해주는 도구임. 
# 데커레이터 패턴은 함수의 객체와 함수를 변경하는 다른 객체의 래핑을 허용한다. 

def C(object):
    @my_decorator
    def method(self):
        @ 메서드 내용

# 위의 코드는 다음과 같다.

def C(object):
    def method(self):
        method = my_decorator(method)

import random
import time

def benchmark(func):
    def wrapper(*args, **kwargs):
        t = time.perf_counter() # 코드 실행시간 측정
        res = func(*args, **kwargs)
        print('{0} {1}'.format(func.__name__, time.perf_counter()-t))
        return res
    return wrapper

@benchmark
def random_tree(n):
    temp = [n for n in range(n)]
    for i in range(n+1):
        temp[random.choice(temp)] = random.choice(temp)
    return temp

if __name__ == '__main__':
    random_tree(1000) # random_tree 0.001533843

# 파이썬에서 일반적으로 사용하는 데커레이터는 @classmethod 와 @static-method이다. 
# 각각 메서드를 클래스와 정적 메서드로 변환 시켜준다. 
# @classmethod는 첫번째 인수로 클래스를 사용하고 @static-method는 첫번째 인수에 self 혹은 클래스가 없다. 

class A(object):
    _hello = True

    def foo(self,x):
        print('foo({0},{1}) 실행'.format(self,x))

    @classmethod
    def class_foo(cls,x):
        print('class_foo({0},{1} 실행: {2}'.format(cls,x,cls._hello))

    @staticmethod
    def static_foo(x):
        print('static_foo{0} 실행'.format(x))

if __name__ == '__main__':
    a = A()
    a.foo(1) # foo(<__main__.A object at 0x10a3e4f10>,1) 실행
    a.class_foo(2) # class_foo(<class '__main__.A'>,2 실행: True
    A.class_foo(2) # class_foo(<class '__main__.A'>,2 실행: True
    a.static_foo(3) # static_foo3 실행
    A.static_foo(3) # static_foo3 실행

# 5.3.2 옵서버 패턴

# 옵서버 패턴은 특정 값을 유지하는 핵심 객체를 갖고, 직렬화된 객체의 복사본을 생성하는 일부 옵서버가 있는 경우 유용하다. 
# 객체의 일대다 의존 관계에서 한 객체의 상태가 변경되면, 그 객체에 종속된 모든 객체에 그 내용을 통지하여 자동으로 상태를 갱신하는 방식. 
# 옵서버 패턴은 @property 데커레이터를 사용하여 구현할 수 있다. 

class C:
    def __init__(self,name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = "{0} >> {1}".format(self._name, new_name)
    
c = C('진')
print(c._name) # 진
print(c.name) # 진
c.name = '아스틴'
print(c.name) # 진 >> 아스틴

# 셋을 사용한 옵서버 패턴 구현

class Subscriber(object):
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print('{0}, {1}'.format(self.name,message))
    
class Publisher(object): 
    def __init__(self):
        self.subscribers = set()

    def register(self, who):
        self.subscribers.add(who)
    
    def unregister(self, who):
        self.subscribers.discard(who)

    def dispatch(self, message):
        for subscriber in self.subscribers:
            subscriber.update(message)

if __name__ == '__main__':
    pub = Publisher()

    astin = Subscriber('아스틴')
    james = Subscriber('제임스')
    jeff = Subscriber('제프')

    pub.register(astin)
    pub.register(james)
    pub.register(jeff)

    pub.dispatch('점심시간입니다.')
    pub.unregister(jeff)
    pub.dispatch('퇴근시간입니다.')

# 제임스, 점심시간입니다.
# 제프, 점심시간입니다.
# 아스틴, 점심시간입니다.
# 제임스, 퇴근시간입니다.
# 아스틴, 퇴근시간입니다.

# 딕셔너리를 사용한 옵서버 패턴 구현 

class SubscriberOne(object):
    def __init__(self,name):
        self.name = name
    
    def update(self, message):
        print('{0}, {1}'.format(self.name,message))
    
class SubscriberTwo(object):
    def __init__(self,name):
        self.name = name
    
    def receive(self, message):
        print('{0}, {1}'.format(self.name,message))

class Publisher(object):
    def __init__(self):
        self.subscribers = dict()

    def register(self, who, callback = None):
        if callback is None:
            callback = getattr(who, 'update')
        self.subscribers[who] = callback
    
    def unregister(self, who):
        del self.subscribers[who]

    def dispatch(self, message):
        for subscriber, callback in self.subscribers.items():
            callback(message)

if __name__ == '__main__':
    pub = Publisher()

    astin = SubscriberOne('아스틴')
    james = SubscriberTwo('제임스')
    jeff = SubscriberOne('제프')

    pub.register(astin, astin.update)
    pub.register(james, james.receive)
    pub.register(jeff)

    pub.dispatch('점심시간입니다.')
    pub.unregister(jeff)
    pub.dispatch('퇴근시간입니다.')

# 아스틴, 점심시간입니다.
# 제임스, 점심시간입니다.
# 제프, 점심시간입니다.
# 아스틴, 퇴근시간입니다.
# 제임스, 퇴근시간입니다.

# 이벤트 기반 옵서버 패턴

class Subscriber(object):
    def __init__(self, name):
        self.name = name

    def update(self, message):
        print('{0}, {1}'.format(self.name,message))

class Publisher(object):
    def __init__(self, events):
        self.subscribers = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self.subscribers[event]
    
    def register(self, event, who, callback = None):
        if callback is None:
            callback = getattr(who, 'update')
        self.get_subscribers(event)[who] = callback

    def unregister(self, who):
        del self.subscribers(event)[who]

    def dispatch(self,event, message):
        for subscriber, callback in self.get_subscribers(event).items():
            callback(message)

if __name__ == '__main__':
    pub = Publisher(['점심','퇴근'])

    astin = Subscriber('아스틴')
    james = Subscriber('제임스')
    jeff = Subscriber('제프')

    pub.register('점심', astin)
    pub.register('퇴근',astin)
    pub.register('퇴근',james)
    pub.register('점심',jeff)

    pub.dispatch('점심','점심시간입니다.')
    pub.dispatch('퇴근','저녁시간입니다.')

# 아스틴, 점심시간입니다.
# 제프, 점심시간입니다.
# 아스틴, 저녁시간입니다.
# 제임스, 저녁시간입니다.

# 5.3.3 싱글턴 패턴

# 초기화된 객체의 인스턴스를 전역에서 사용하기 위해서는 싱글턴 패턴을 사용한다. 
# 이 객체의 인스턴스는 하나만 존재 
# 파이썬에는 private 접근 제한이 없기 때문에 __new__() 클래스 메서드를 가지고 하나의 인스턴스만 생성되도록 구현해야된다. 

class SinEx:
    _sing = None

    def __new__(self, *args, **kwargs):
        if not self._sing:
            self._sing = super(SinEx, self).__new__(self,*args,**kwargs)
        return self._sing

x = SinEx()
print(x)
y = SinEx()
print(x == y)
print(y)

# <__main__.SinEx object at 0x1072fdc10>
# True
# <__main__.SinEx object at 0x1072fdc10>
# 두 변수의 객체 주소가 같기 때문에 두 객체는 같다고 볼 수 있다. 