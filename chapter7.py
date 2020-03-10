# 7. 추상 데이터 타입

# 추상 데이터 타입은 유사한 동작을 가진 자료구조의 클래스에 대한 수학적 모델을 가리킨다.
# 자료구조는 크게 배열 기반의 연속 방식과 포인터 기반의 연결 방식으로 분류한다. 

# 7.1 스택

# 스택은 배열의 끝에서만 데이터를 접근할 수 있는 선형 자료구조. 
# 배열 인덱스 접근이 제한되며 후입선출 구조다. 시간 복잡도는 O(1)이다. (나중에 들어온 것이 먼저 나가는 구조)

# push: 스택 맨 끝에 항목을 삽입한다.
# pop: 스택 맨 끝 항목을 반환하는 동시에 제거한다. 
# top/peek: 스택 맨 끝 항목을 조회한다.
# empty: 스택이 비어 있는지 확인한다.
# size: 스택 크기를 확인한다.

# 파이썬에서는 append()와 pop()으로 스택 구현이 가능하다. 

class Stack(object):
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return not bool(self.items)

    def push(self, value):
        self.items.append(value)
    
    def pop(self):
        value = self.items.pop()
        if value is not None:
            return value
        else: 
            print('Stack is Empty')

    def size(self):
        return len(self.items)

    def peek(self):
        if self.items:
            return self.items[-1]
        else:
            print('Stack is Empty')

    def __repr__(self):
        return repr(self.items)

if __name__ == '__main__':
    stack = Stack()
    print('is stack empty? {0}'.format(stack.isEmpty()))
    print('add number 0~9 to stack.')
    for i in range(10):
        stack.push(i)
    print(stack.size())
    print(stack.peek())
    print(stack.pop())
    print(stack.pop())
    print(stack.peek())
    print(stack.isEmpty())
    print(stack)

# is stack empty? True
# add number 0~9 to stack.
# 10
# 9
# 9
# 8
# 7
# False
# [0, 1, 2, 3, 4, 5, 6, 7]

# 노드의 컨테이너로 스택을 구현. 

class Node(object):
    def __init__(self,value=None,pointer=None):
        self.value = value
        self.pointer = pointer

class Stack():
    def __init__(self):
        self.head = None
        self.count = 0 
    
    def isEmpty(self):
        return not bool(self.head)

    def push(self, item):
        self.head = Node(item,self.head)
        self.count += 1

    def pop(self):
        if self.count > 0 and self.head:
            node = self.head
            self.head = node.pointer
            self.count -= 1
            return node.value
        else:
            print('stack is empty')

    def peek(self):
        if self.count > 0 and self.head:
            return self.head.value
        else:
            print('stack is empty')

    def size(self):
        return self.count

    def _printList(self):
        node = self.head
        while node:
            print(node.value, end = ' ')
            node = node.pointer
        print()

if __name__ == '__main__':
    stack = Stack()
    print('is stack empty? {0}'.format(stack.isEmpty()))
    print('add number 0~9 to stack.')
    for i in range(10):
        stack.push(i)
    print(stack.size())
    print(stack.peek())
    print(stack.pop())
    print(stack.peek())
    print(stack.isEmpty())
    stack._printList()

# is stack empty? True
# add number 0~9 to stack.
# 10
# 9
# 9
# 8
# False
# 7 6 5 4 3 2 1 0 

# 스택은 깊이 우선 탐색(DFS)에서 유용하게 사용되고 있음. 

# 7.2 큐

# 큐는 스택과 다르게 항목이 들어온 순서대로 접근 가능하다. 
# 먼저 들어온 데이터가 먼저 나가는 선입선출 구조다. 
# 큐도 스택과 마찬가지로 배열의 인덱스 접근이 제한된다. 
# 시간복잡도는 O(1)이다.

# enqueue: 큐 뒤쪽에 항목을 삽입한다. 
# dequeue: 큐 앞쪽의 항목을 반환하고, 제거한다. 
# peek/front: 큐 앞쪽의 항목을 조회한다.
# empty: 큐가 비어 있는지 확인한다. 
# size: 큐의 크기를 확인한다. 

class Queue(object):
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return not bool(self.items)

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        value = self.items.pop()
        if value is not None:
            return value
        else: 
            print('queue is empty')

    def size(self):
        return len(self.items)

    def peek(self):
        if self.items:
            return self.items[-1]
        else:
            print('queue is empty')

    def __repr__(self):
        return repr(self.items)

if __name__ == '__main__':
    queue = Queue()
    print('is queue empty? {0}'.format(queue.isEmpty()))
    print('add number 0~9 to queue.')
    for i in range(10):
        queue.enqueue(i)
    print(queue.size())
    print(queue.peek())
    print(queue.dequeue())
    print(queue.peek())
    print(queue.isEmpty())
    print(queue)

# is queue empty? True
# add number 0~9 to queue.
# 10
# 0
# 0
# 1
# False
# [9, 8, 7, 6, 5, 4, 3, 2, 1]

# insert를 사용하게 되면 시간복잡도가 O(n)이기 때문에 비효율 적이다. 
# 두개의 스택을 사용하게 되면 효율적인 큐를 만들 수 있다. 

class Queue(object):
    def __init__(self):
        self.in_stack = []
        self.out_stack = []
    
    def _transfer(self):
        while self.in_stack:
            self.out_stack.append(self.in_stack.pop())
    
    def enqueue(self, item):
        return self.in_stack.append(item)

    def dequeue(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return self.out_stack.pop()
        else: 
            print('queue is empty')

    def size(self):
        return len(self.in_stack) + len(self.out_stack)

    def peek(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return self.out_stack[-1]
        else:
            print('queue is empty')

    def __repr__(self):
        if not self.out_stack:
            self._transfer()
        if self.out_stack:
            return repr(self.out_stack)
        else: 
            print('queue is empty')
    
    def isEmpty(self):
        return not bool((self.in_stack) or (self.out_stack))

if __name__ == '__main__':
    queue = Queue()
    print('is queue empty? {0}'.format(queue.isEmpty()))
    print('add number 0~9 to queue.')
    for i in range(10):
        queue.enqueue(i)
    print(queue.size())
    print(queue.peek())
    print(queue.dequeue())
    print(queue.peek())
    print(queue.isEmpty())
    print(queue)

# is queue empty? True
# add number 0~9 to queue.
# 10
# 0
# 0
# 1
# False
# [9, 8, 7, 6, 5, 4, 3, 2, 1]

# 노드의 컨테이너로 큐를 구현.

class Node(object):
    def __init__(self, value = None, pointer = None):
        self.value = value
        self.pointer = None
    
class LinkedQueue(object):
    def __init__(self):
        self.head = None
        self.tail = None
        self.count = 0

    def isEmpty(self):
        return not bool(self.head)

    def dequeue(self):
        if self.head:
            value = self.head.value
            self.head = self.head.pointer
            self.count -= 1
            return value
        else: 
            print('queue is empty')

    def enqueue(self,value):
        node = Node(value)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            if self.tail:
                self.tail.pointer = node
            self.tail = node
        self.count += 1

    def size(self):
        return self.count
    
    def peek(self):
        return self.head.value
    
    def print(self):
        node = self.head
        while node:
            print(node.value, end = ' ')
            node = node.pointer
        print()

if __name__ == '__main__':
    queue = LinkedQueue()
    print('is queue empty? {0}'.format(queue.isEmpty()))
    print('add number 0~9 to queue.')
    for i in range(10):
        queue.enqueue(i)
    queue.print()
    print(queue.size())
    print(queue.peek())
    print(queue.dequeue())
    print(queue.peek())
    queue.print()

# is queue empty? True
# add number 0~9 to queue.
# 0 1 2 3 4 5 6 7 8 9 
# 10
# 0
# 0
# 1
# 1 2 3 4 5 6 7 8 9 

# 큐는 너비 우선 탐색(BFS)에서 사용된다. 

# 7.3 데크(deque)

# 데크는 스택과 큐의 결합체.
# 양쪽 끝에서 항목의 조회, 삽입, 삭제가 가능하다. 

from queue import Queue

class Deque(Queue):
    def enqueue_back(self,item):
        self.items.append(item)

    def dequeue_front(self):
        value = self.items.pop(0)
        if value is not None:
            return value
        else:
            print('deque is empty')

# 끝이 아닌 다른 위치에 있는 항목을 삽입하거나 제거할 때는 비효율적임. collections 패키지의 deque 모듈로 해결할 수 있다. 

from collections import deque

q = deque(['버피','잰더','윌로'])
print(q)
q.append('자일스')
print(q)
print(q.popleft())
print(q.pop())
print(q)
q.appendleft('엔젤')
print(q)

# deque(['버피', '잰더', '윌로'])
# deque(['버피', '잰더', '윌로', '자일스'])
# 버피
# 자일스
# deque(['잰더', '윌로'])
# deque(['엔젤', '잰더', '윌로'])

# deque 모듈을 사용하게 되면 q = deque(maxlen=4) 같은 방식으로 테크의 크기를 지정할 수 있다. 
# rotate(n) 메서드는 n이 양수면 오른쪽, 음수면 왼쪽으로 n만큼 시프트한다. 
q.rotate(1)
print(q)
q.rotate(2)
print(q)
q.rotate(4)
print(q)
q.rotate(-1)
print(q)
q.rotate(-2)
print(q)

# deque(['윌로', '엔젤', '잰더'])
# deque(['엔젤', '잰더', '윌로'])
# deque(['윌로', '엔젤', '잰더'])
# deque(['엔젤', '잰더', '윌로'])
# deque(['윌로', '엔젤', '잰더'])

# deque는 동적 배열이 아닌 이중 연결 리스트를 기반으로 한다. 

# 7.4 우선순위 큐와 힙

# 우선순위 큐는 일반 스택, 큐와 비슷한 추상 데이터 타입이지만, 각 항목마다 연관된 우선순위가 존재한다. 
# 두 항목의 우선순위가 같으면 큐의 순서를 따른다. 
# 우선순위 큐는 힙을 사용해서 구현한다. 

# 7.4.1 힙

# 힙은 각 노드가 휘위 노드보다 작은(또는 큰) 이진 트리다. 
# 균형 트리의 모양이 수정될 때 다시 이를 균형 트리로 만드는 시간복잡도는 O(logn)이다.
# 리스트에서 가장 작은(또는 가장 큰) 요소에 반복적으로 접근하는 프로그램에 유용하다.
# 최소(또는 최대) 힙을 사용하면 가장 작은 (또는 가장 큰) 요소를 처리하는 시간복잡도는 O(1)이다. 
# 그 외 조회, 추가 수정을 처리하는 시간복잡도는 O(logn)이다.

# 7.4.2 heapq 모듈

# heapq 모듈은 효율적으로 시퀀스를 힙으로 유지하면서 항목을 삽입하고 삭제하는 함수를 제공한다. 
# heapify()를 사용하면 O(n) 시간에 리스트를 힙으로 변환할 수 있다. 

import heapq

list1 = [4,6,8,1]
heapq.heapify(list1)
print(list1) # [1, 4, 8, 6]

# 항목을 삽입할 때는 heappush(heap, item)을 쓴다. 

import heapq
h = []
heapq.heappush(h,(1,'food'))
heapq.heappush(h, (2,'have fun'))
heapq.heappush(h, (3,'work'))
heapq.heappush(h, (4,'study'))
print(h)

# heapq.heappop(heap) 함수는 힙에서 가장 작은 항목을 제거하고 반환한다. 
import heapq

heapq.heappop(list1)
print(list1) # [4, 8, 6]

# heappushpop()은 새 항목을 추가한 후 가장 작은 항목을 제거하고 반환하는 기능. 
# heapreplace()은 힙의 가장 작은 항목을 반환한 후, 새로운 항목을 추가하는 기능이다.
# heappop()이나 heappush()보다는 위의 두개가 좀 더 효율적이다. 
# 힙의 속성을 사용하면 많은 연산을 할 수 있다.
# heapq.merge(*iterables)는 여러 개의 정렬된 반복 가능한 객체를 병합하여 하나의 정렬된 결과의 이터레이터를 반환한다.
import heapq
for x in heapq.merge([1,3,5],[2,4,6]):
    print(x)

    # 1~6까지 순서대로 반환된다. 

# heapq.nlargest(n,iterable[,key])와 heapq.nsmallest(n,iterable[,key])는 데이터에서 n개의 가장 큰 요소와 가장 작은 요소가 있는 리스트를 반환한다. 

# 7.4.3 최대 힙 구하기

# [3,2,5,1,7,8,2]라는 리스트가 있을 때, 이진트리로 결과를 나타내게 되면 3의 자식은 2와 5가 되는 것이고, 
# 2의 자식은 1과 7, 그리고 5의 자식은 8과 2가 된다. 
# 노드 i의 왼쪽 자식 노드의 인덱스는 (i*2) + 1이고, i의 오른쪽 자식 노드의 인덱스는 (i*2) + 2다. 

class Heapify(object):
    def __init__(self, data = None):
        self.data = data or []
        for i in range(len(data)//2,-1,-1):
            self.__max_heapify__(i)

    def __repr__(self):
        return repr(self.data)

    def parent(self, i):
        if i & 1: # 왼쪽(오른쪽) 시프트 연산자. 변수의 값을 왼쪽(오른쪽)으로 지정된 비트 수 만큼 이동
            return i >> 1
        
    def left_child(self, i):
        return (i << 1) + 1

    def right_child(self, i):
        return (i << 2) + 2 
    
    def __max_heapify__(self, i):
        largest = i # 현재 노드
        left = self.left_child(i)
        right = self.right_child(i)
        n = len(self.data)

        # 왼쪽 자식
        largest = (left < n and self.data[left] > self.data[i]) and left or i 
        # 오른쪽 자식
        largest = (right < n and self.data[right] > self.data[i]) and right or i

        # 현재 노드가 자식들보다 크다면 Pass, 자식이 크다면 Swap
        if i is not largest:
            self.data[i], self.data[largest] = self.data[largest], self.data[i]

            self.__max_heapify__(largest)
    
    def extract_max(self):
        n = len(self.data)
        max_element = self.data[0]

        # 첫 번째 노드에 마지막 노드를 삽입
        self.data[0] = self.data[n-1]
        self.data = self.data[:n - 1]
        self.__max_heapify__(0)
        return max_element

    def insert(self, item):
        i = len(self.data)
        self.data.append(item)
        while (i != 0) and item > self.data[self.parent(i)]:
            print(self.data)
            self.data[i] = self.data[self.parent(i)]
            i = self.parent(i)
        self.data[i] = item 

def test_heapify():
    l1 = [3,2,5,1,7,8,2]
    h = Heapify(l1)
    assert(h.extract_max() == 8)
    print('테스트 통과')

if __name__ == '__main__':
    test_heapify() 

# 7.4.4 우선순위 큐 구현하기

# 숫자가 클수록 우선순위가 높게 만들어주면 된다. 

import heapq

class PriorityQueue(object):
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue,(-priority, self._index, item))
    
    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Item:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Item({0!r})".format(self.name)
    
def test_priority_queue():
    # push와 pop은 모두 O(logn)이다
    q = PriorityQueue()
    q.push(Item('test1'), 1)
    q.push(Item('test2'), 4)
    q.push(Item('test2'), 3)
    assert(str(q.pop()) == "Item('test2')")
    print('테스트 통과')

if __name__ == '__main__':
    test_priority_queue()