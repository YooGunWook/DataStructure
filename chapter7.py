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

# 7.5 연결 리스트 

# 연결 리스트는 값과 다음 노드에 대한 포인터가 포함된 노드로 이루어진 선형 리스트다. 
# 마지막 노드는 null 값을 가지고 있다. 
# 연결 리스트로 스택과 큐를 구현할 수 있다. 

class Node(object):
    def __init__(self, value = None, pointer = None):
        self.value = value
        self.pointer = pointer 

    def getData(self):
        return self.value
    
    def getNext(self):
        return self.pointer

    def setData(self, newdata):
        self.value = newdata

    def setNext(self, newpointer):
        self.pointer = newpointer
    
if __name__ == '__main__':
    L = Node('a', Node('b',Node('c',Node('d'))))
    assert(L.pointer.pointer.value == 'c')

    print(L.getData())
    print(L.getNext().getData())
    L.setData('aa')
    L.setNext(Node('e'))
    print(L.getData())
    print(L.getNext().getData())
# a
# b
# aa
# e

# 후입선출 연결 리스트 구현

class Node(object):
    def __init__(self, value = None, pointer = None):
        self.value = value
        self.pointer = pointer 

    def getData(self):
        return self.value
    
    def getNext(self):
        return self.pointer

    def setData(self, newdata):
        self.value = newdata

    def setNext(self, newpointer):
        self.pointer = newpointer

class LinkedListLIFO(object):
    def __init__(self):
        self.head = None
        self.length = 0

    # 헤드부터 각 노드 값을 출력한다. 
    def _printList(self):
        node = self.head
        while node:
            print(node.value, end = " ")
            node = node.pointer
        print()

    # 이전 노드(prev)를 기반으로 노드를 삭제한다. 
    def _delete(self, prev, node):
        self.length -= 1
        if not prev:
            self.head = node.pointer
        else:
            prev.pointer = node.pointer

    # 새 노드를 추가한다. 다음 노드로 헤드를 가리키고, 헤드는 새 노드를 가리킨다.     
    def _add(self, value):
        self.length += 1
        self.head = Node(value, self.head)

    # 인덱스로 노드를 찾는다. 
    def _find(self, index):
        prev = None
        node = self.head
        i = 0
        while node and i < index:
            prev = node
            node = node.pointer
            i += 1
        return node, prev, i 

    # 값으로 노드를 찾는다.
    def _find_by_value(self,value):
        prev = None
        node = self.head
        found = False
        while node and not found:
            if node.value == value:
                found = True
            else:
                prev = node
                node = node.pointer
        return node, prev, found

    # 인덱스에 해당하는 노드를 찾아서 삭제한다. 
    def deleteNode(self, index):
        node, prev, i = self._find(index)
        if index == i:
            self._delete(prev,node)
        else:
            print(f"인덱스 {index}에 해당하는 노드가 없습니다. ")
        
    # 값에 해당하는 노드를 찾아서 삭제한다. 
    def deleteNodeByValue(self, value):
        node, prev, found = self._find_by_value(value)
        if found:
            self._delete(prev,node)
        else:
            print(f"값{value}에 해당하는 노드가 없습니다.")

if __name__ == '__main__':
    ll = LinkedListLIFO()
    for i in range(1,5):
        ll._add(i)
    print('연결 리스트 출력:')
    ll._printList()
    print('인덱스 2인 노드 삭제 후, 연결 리스트 출력:')
    ll.deleteNode(2)
    ll._printList()
    print('값이 3인 노드 삭제 후, 연결 리스트 출력:')
    ll.deleteNodeByValue(3)
    ll._printList()
    print('값이 15인 노드 추가 후, 연결 리스트 출력:')
    ll._add(15)
    ll._printList()
    print('모든 노드 삭제 후, 연결 리스트 출력:')
    for i in range(ll.length-1,-1,-1):
        ll.deleteNode(i)
    ll._printList()

# 연결 리스트 출력:
# 4 3 2 1 
# 인덱스 2인 노드 삭제 후, 연결 리스트 출력:
# 4 3 1 
# 값이 3인 노드 삭제 후, 연결 리스트 출력:
# 4 1 
# 값이 15인 노드 추가 후, 연결 리스트 출력:
# 15 4 1 
# 모든 노드 삭제 후, 연결 리스트 출력:

# 선입선출 형식의 연결 리스트 구현

class Node(object):
    def __init__(self, value = None, pointer = None):
        self.value = value
        self.pointer = pointer 

    def getData(self):
        return self.value
    
    def getNext(self):
        return self.pointer

    def setData(self, newdata):
        self.value = newdata

    def setNext(self, newpointer):
        self.pointer = newpointer

class LinkedListFIFO(object):
    def __init__(self):
        self.head = None #머리부분
        self.length = 0
        self.tail = None #꼬리부분

    # 헤드부터 각 노드의 값을 출력한다.
    def _printList(self):
        node = self.head
        while node:
            print(node.value, end = " ")
            node = node.pointer
        print()
    
    # 첫번째 위치에 노드를 추가한다.
    def _addFirst(self,value):
        self.length = 1
        node = Node(value)
        self.head = node
        self.tail = node
    
    # 첫번째 위치에 노드를 삭제한다. 
    def _deleteFirst(self):
        self.length = 0
        self.head = None
        self.tail = None
        print('연결 리스트가 비었습니다.')

    # 새 노드를 추가한다. 테일이 있다면, 테일의 다음 노드는 새 노드를 가리키고, 테일은 새 노드를 가리킨다. 
    def _add(self,value):
        self.length += 1
        node = Node(value)
        if self.tail:
            self.tail.pointer = node
        self.tail = node
    
    # 새 노드를 추가한다.
    def addNode(self, value):
        if not self.head:
            self._addFirst(value)
        else:
            self._add(value)

    # 인덱스로 노드를 찾는다. 
    def _find(self, index):
        prev = None
        node = self.head
        i = 0
        while node and i < index:
            prev = node
            node = node.pointer
            i += 1
        return node, prev, i 

    # 값으로 노드를 찾는다. 
    def _find_by_value(self,value):
        prev = None
        node = self.head
        found = False
        while node and not found:
            if node.value == value:
                found = True
            else:
                prev = node
                node = node.pointer
        return node, prev, found
    
    # 인덱스에 해당하는 노드를 삭제한다. 
    def deleteNode(self, index):
        if not self.head or not self.head.pointer:
            self._deleteFirst()
        else:
            node, prev, i = self._find(index)
            if i == index and node:
                self.length -= 1
                if i == 0 or not prev:
                    self.head = node.pointer
                    self.tail = node.pointer
                else:
                    prev.pointer = node.pointer
            else:
                print('값 {0}에 해당하는 노드가 없습니다.'.format(index))

    # 값에 해당하는 노드를 삭제한다. 
    def deleteNodeByValue(self, value):
        if not self.head or not self.head.pointer:
            self._deleteFirst()
        else:
            node, prev, i = self._find(index)
            if node and node.value == value:
                self.length -= 1
                if i == 0 or not prev:
                    self.head = node.pointer
                    self.tail = node.pointer
                else: 
                    prev.ponter = node.pointer
            else:
                print('값 {0}에 해당하는 노드가 없습니다.'.format(value))

if __name__ == '__main__':
    ll = LinkedListFIFO()
    for i in range(1,5):
        ll.addNode(i)
    print('연결 리스트 출력:')
    ll._printList()
    print('인덱스 2인 노드 삭제 후, 연결 리스트 출력:')
    ll.deleteNode(2)
    ll._printList()
    print('값이 15인 노드 추가 후, 연결 리스트 출력:')
    ll.addNode(15)
    ll._printList()
    print('모든 노드 삭제 후, 연결 리스트 출력:')
    for i in range(ll.length-1,-1,-1):
        ll.deleteNode(i)
    ll._printList()

#연결 리스트 출력:
#1 2 3 4 
#인덱스 2인 노드 삭제 후, 연결 리스트 출력:
#1 2 4 
#값이 15인 노드 추가 후, 연결 리스트 출력:
#1 2 4 15 
#모든 노드 삭제 후, 연결 리스트 출력:
#연결 리스트가 비었습니다.

# 연결 리스트의 크기는 동적일 수 있다. 따라서 런타임에 저장할 항목의 수를 알 수 없을 때 유용하다. 
# 연결 리스트의 삽입 시간복잡도는 O(1)이다. 
# 검색및 삭제의 시간복잡도는 O(n)이다. (순차적으로 항목을 검색하기 때문)
# 뒤부터 순회하거나 정렬하는 최악의 경우는 O(n^2)이다
# 어떤 노드의 포인터를 알고 있을 때 그 노드를 삭제하면 시간복잡도는 O(1)이 될 수 있다. 
# -> 해당 노드의 값에 다음 노드의 값을 할당하고, 해당 노드의 포인터는 다음 다음의 노드를 가리키게 하면 되기 때문이다. 

# 삭제 코드
if node.pointer is not None:
    node.value = node.pointer.value
    node.pointer = node.pointer.pointer
else:
    node = None

# 7.6 해시 테이블

# 해시 테이블은 키를 값에 연결하여, 하나의 키가 0 또는 1개의 값과 연관된다. 
# 각 키는 해시 함수를 계산할 수 있어야한다. 
# 해시 테이블은 해시 버킷의 배열로 구성됨. 
# 두개의 키가 동이ㅣㄹ한 버킷에 해시될 때, 해시 충돌이 발생한다. 
# 이를 처리하기 위해서는 각 버킷에 대해 키-값 쌍의 연결리스트를 저장하는 것이다. 
# 해시테이블의 조회, 삽입 삭제의 시간 복잡도는 O(1)이다. 최악의 경우 동일한 버킷으로 해시된다면(해시충돌), 각 작업의 시간 복잡도는 O(n)이다.


class Node(object):
    def __init__(self, value = None, pointer = None):
        self.value = value
        self.pointer = pointer 

    def getData(self):
        return self.value
    
    def getNext(self):
        return self.pointer

    def setData(self, newdata):
        self.value = newdata

    def setNext(self, newpointer):
        self.pointer = newpointer

class LinkedListFIFO(object):
    def __init__(self):
        self.head = None #머리부분
        self.length = 0
        self.tail = None #꼬리부분

    # 헤드부터 각 노드의 값을 출력한다.
    def _printList(self):
        node = self.head
        while node:
            print(node.value, end = " ")
            node = node.pointer
        print()
    
    # 첫번째 위치에 노드를 추가한다.
    def _addFirst(self,value):
        self.length = 1
        node = Node(value)
        self.head = node
        self.tail = node
    
    # 첫번째 위치에 노드를 삭제한다. 
    def _deleteFirst(self):
        self.length = 0
        self.head = None
        self.tail = None
        print('연결 리스트가 비었습니다.')

    # 새 노드를 추가한다. 테일이 있다면, 테일의 다음 노드는 새 노드를 가리키고, 테일은 새 노드를 가리킨다. 
    def _add(self,value):
        self.length += 1
        node = Node(value)
        if self.tail:
            self.tail.pointer = node
        self.tail = node
    
    # 새 노드를 추가한다.
    def addNode(self, value):
        if not self.head:
            self._addFirst(value)
        else:
            self._add(value)

    # 인덱스로 노드를 찾는다. 
    def _find(self, index):
        prev = None
        node = self.head
        i = 0
        while node and i < index:
            prev = node
            node = node.pointer
            i += 1
        return node, prev, i 

    # 값으로 노드를 찾는다. 
    def _find_by_value(self,value):
        prev = None
        node = self.head
        found = False
        while node and not found:
            if node.value == value:
                found = True
            else:
                prev = node
                node = node.pointer
        return node, prev, found
    
    # 인덱스에 해당하는 노드를 삭제한다. 
    def deleteNode(self, index):
        if not self.head or not self.head.pointer:
            self._deleteFirst()
        else:
            node, prev, i = self._find(index)
            if i == index and node:
                self.length -= 1
                if i == 0 or not prev:
                    self.head = node.pointer
                    self.tail = node.pointer
                else:
                    prev.pointer = node.pointer
            else:
                print('값 {0}에 해당하는 노드가 없습니다.'.format(index))

    # 값에 해당하는 노드를 삭제한다. 
    def deleteNodeByValue(self, value):
        if not self.head or not self.head.pointer:
            self._deleteFirst()
        else:
            node, prev, i = self._find(index)
            if node and node.value == value:
                self.length -= 1
                if i == 0 or not prev:
                    self.head = node.pointer
                    self.tail = node.pointer
                else: 
                    prev.ponter = node.pointer
            else:
                print('값 {0}에 해당하는 노드가 없습니다.'.format(value))

class HashTableLL(object):
    def __init__(self,size):
        self.size = size
        self.slots = []
        self._createHashTable()
    
    def _createHashTable(self):
        for i in range(self.size):
            self.slots.append(LinkedListFIFO())

    def _find(self, item):
        return item % self.size
    
    def _add(self, item):
        index = self._find(item)
        self.slots[index].addNode(item)
    
    def _delete(self, item):
        index = self._find(item)
        self.slots[index].deleteNodeByValue(item)

    def _print(self):
        for i in range(self.size):
            print('슬롯 {0}:'.format(i))
            self.slots[i]_printList()


def test_hash_table():
    H1 = HashTableLL(3)
    for i in range(0,20):
        H1._add(i)
    H1._print()
    print('\n항목 0,1,2를 삭제합니다.')
    H1._delete(0)
    H1._delete(1)
    H1._delete(2)
    H1.print()

if __name__ == '__main__':
    test_hash_table()

# 7.7 연습문제

# 7.7.1 스택
# 문자열 반전
list1 = []
list2 = []
str1 = '버피는 천사다.'
for i in str1:
    list1.append(i)
#for i in list1:
 #   a = list1.pop()
  #  list2.append(a)
#print(list2)

while list1:
    a = list1.pop()
    list2.append(a)
print(list2)
list2
print(''.join(list2))
# .다사천 는피버

# 괄호의 짝 확인하기
list1 = []
list2 = []
str1 = '((()))'
str2 = '(()'
count1 = 0
count2 = 0


for i in str1:
    if i == '(':
        count1 += 1
    if i == ')':
        count2 += 1

print(count1)
print(count2)

print(count1 == count2) # True


count1 = 0
count2 = 0
for i in str2:
    if i == '(':
        count1 += 1
    if i == ')':
        count2 += 1
print(count1 == count2) # False

# 10진수를 2진수로 변환

list1  = [] 
decum = 9
str_pop = ''

while decum > 0:
    dig = decum % 2
    decum = decum //2
    list1.append(dig)

while list1:
    str_pop += str(list1.pop())

print(str_pop) # 1001

# 스택에서 최솟값 O(1)로 조회하기 

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

class NodeWithMin(object):
    def __init__(self, value = None, minimum = None):
        self.value = value
        self.minimum = minimum

class StackMin(Stack):
    def __init__(self):
        self.items = []
        self.minimum = None

    def push(self, value):
        if self.isEmpty() or self.minimum > value:
            self.minimum = value
        self.items.append(NodeWithMin(value, self.minimum))

    def peek(self):
        return self.items[-1].value
    
    def peekMinimum(self):
        return self.items[-1].minimum

    def pop(self):
        item = self.items.pop()
        if item:
            if item.value  == self.minimum:
                self.minimum = self.peekMinimum()
            return item.value
        else:
            print('Stack is empty')

    def __repr__(self):
        aux = []
        for i in self.items:
            aux.append(i.value)
        return repr(aux)

if __name__ == '__main__':
    stack = StackMin()
    print('스택이 비었나요? {0}'.format(stack.isEmpty()))
    print('스택에 10~1 과 1~4를 추가한다.')
    for i in range(10,0,-1):
        stack.push(i)
    for i in range(1,5):
        stack.push(i)
    print(stack)
    print('스택 크기 : {0}'.format(stack.size()))
    print('peek : {0}'.format(stack.peek()))
    print('peekMinimum: {0}'.format(stack.peekMinimum()))
    print('pop: {0}'.format(stack.pop()))
    print('peek: {0}'.format(stack.peek()))
    print('peekMinimum: {0}'.format(stack.peekMinimum()))
    print('스택이 비었나요? {0}'.format(stack.isEmpty()))
    print(stack)

# 스택이 비었나요? True
# 스택에 10~1 과 1~4를 추가한다.
# [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4]
# 스택 크기 : 14
# peek : 4
# peekMinimum: 1
# pop: 4
# peek: 3
# peekMinimum: 1
# 스택이 비었나요? False
# [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3]

# 스택 집합
# 스택에 용량이 정해져 있다면, 용량이 초과되면 새로운 스택을 만들어야 한다. 

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

class SetOfStacks(Stack):
    def __init__(self, capacity = 4):
        self.setofstacks = []
        self.items = []
        self.capactiy = capacity
    
    def push(self, value):
        if self.size() >= self.capactiy:
            self.setofstacks.append(self.items)
            self.items = []
        self.items.append(value)

    def pop(self):
        value = self.items.pop()
        if self.isEmpty() and self.setofstacks:
            self.items = self.setofstacks.pop()
        return value

    def sizeStack(self):
        return len(self.setofstacks) * self.capactiy + self.size()

    def __repr__(self):
        aux = []
        for s in self.setofstacks:
            aux.extend(s)
        aux.extend(self.items)
        return repr(aux)

if __name__ =='__main__':
    capacity = 5
    stack = SetOfStacks(capacity)
    print('스택이 비었나요? {0}'.format(stack.isEmpty()))
    print('스택에 0~9를 추가한다.')
    for i in range(10):
        stack.push(i)
    print(stack)
    print('스택 크기 : {0}'.format(stack.sizeStack()))
    print('peek : {0}'.format(stack.peek()))
    print('pop: {0}'.format(stack.pop()))
    print('peek: {0}'.format(stack.peek()))
    print('스택이 비었나요? {0}'.format(stack.isEmpty()))
    print(stack)

# 스택이 비었나요? True
# 스택에 0~9를 추가한다.
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 스택 크기 : 10
# peek : 9
# pop: 9
# peek: 8
# 스택이 비었나요? False
# [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 7.7.2 큐

# 데크와 회문
import string
import collections

strip = string.whitespace + string.punctuation + "\"'"

def palindrome_checker_with_deque(str1):
    d2 = collections.deque()

    for s in str1.lower():
        if s not in strip:
            d2.append(s)

    eq2 = True
    while len(d2) > 1 and eq2:
        if d2.pop() != d2.popleft():
            eq2 = False
    return eq2

if __name__ == '__main__':
    str1 = 'Madam Im Adam'
    str2 = 'Buffy is a Slayer'
    print(palindrome_checker_with_deque(str1)) # True
    print(palindrome_checker_with_deque(str2)) # False

# 큐와 동물 보호소 

# 개와 고양이를 입양(enqueue) 했다가 다시 출양(dequeue)할 수 있는 동물 보호소를 큐로 구현. 

class Node(object):
    def __init__(self,animalName = None, animalKind = None, pointer = None):
        self.animalName = animalName
        self.animalKind = animalKind
        self.pointer = pointer
        self.timestamp = 0

class AnimalShelter(object):
    def __init__(self):
        self.headCat = None
        self.headDog = None
        self.tailCat = None
        self.tailDog = None
        self.animalNumber = 0

    def enqueue(self, animalName, animalKind):
        self.animalNumber += 1
        newAnimal = Node(animalName, animalKind)
        newAnimal.timestamp = self.animalNumber

        if animalKind == 'cat':
            if not self.headCat:
                self.headCat = newAnimal
            if self.tailCat:
                self.tailCat.pointer = newAnimal
            self.tailCat = newAnimal

        elif animalKind == 'dog':
            if not self.headDog:
                self.headDog = newAnimal
            if self.tailDog:
                self.tailDog.pointer = newAnimal
            self.tailDog = newAnimal
        
    def dequeueDog(self):
        if self.headDog:
            newAnimal = self.headDog
            self.headDog = newAnimal.pointer
            return str(newAnimal.animalName)
        else: 
            print('개가 없습니다.')
    
    def dequeueCat(self):
        if self.headCat:
            newAnimal = self.headCat
            self.headCat = newAnimal.pointer
            return str(newAnimal.animalName)
        else: 
            print('고양이가 없습니다.')
    
    def dequeueAny(self):
        if self.headCat and not self.headDog:
            return self.dequeueCat()
        elif self.headDog and not self.headCat:
            return self.dequeueDog()
        elif self.headDog and self.headCat:
            if self.headDog.timestamp < self.headCat.timestamp:
                return self.dequeueDog()
            else:
                return self.dequeueCat()
        else:
            print('동물이 없습니다.')

    def _print(self):
        print('고양이:')
        cats = self.headCat
        while cats:
            print('\t{0}'.format(cats.animalName))
            cats = cats.pointer
        print('개:')
        dogs = self.headDog
        while dogs:
            print('\t{0}'.format(dogs.animalName))
            dogs = dogs.pointer

    if __name__ == '__main__':
        qs = AnimalShelter()
        qs.enqueue('밥','cat')
        qs.enqueue('마마','cat')
        qs.enqueue('요다','dog')
        qs.enqueue('울프','dog')
        qs._print()

        print('하나의 개와 하나의 고양이 deque 실행')
        qs.dequeueDog()
        qs.dequeueCat()
        qs._print()

# 7.7.3 우선순위 큐와 힙

import heapq

def find_N_largest_items_seq(seq,N):
    return heapq.nlargest(N,seq)

def find_N_smallest_items_seq(seq,N):
    return heapq.nsmallest(N,seq)

def find_smallest_items_seq_heap(seq):
    heapq.heapify(seq)
    return heapq.heappop(seq)

def find_smallest_items_seq(seq):
    return min(seq)

def find_N_smallest_items_seq_sorted(seq,N):
    return sorted(seq)[:N]

def find_N_largest_items_seq_sorted(seq,N):
    return sorted(seq)[len(seq)-N:]

def test_find_N_largest_smallest_items_seq():
    seq = [1,3,2,8,6,10,9]
    N = 3
    assert(find_N_largest_items_seq(seq,N) == [10,9,8])
    assert(find_N_largest_items_seq_sorted(seq,N) == [8,9,10])
    assert(find_N_smallest_items_seq(seq,N) == [1,2,3])
    assert(find_N_smallest _items_seq_sorted(seq,N) == [1,2,3])
    assert(find_smallest_items_seq(seq) == 1)
    assert(find_smallest_items_seq_heap(seq) == 1)

    print('테스트 통과')

if __name__ == '__main__':
    test_find_N_largest_smallest_items_seq()

import heapq

def merge_sorted_seqs(seq1, seq2):
    result = []
    for c in heapq.merge(seq1,seq2):
        result.append(c)
    return result

def test_merge_sorted():
    seq1 = [1,2,3,8,9,10]
    seq2 = [2,3,4,5,6,7,9]
    seq3 = seq1 + seq2
    assert(merge_sorted_seqs(seq1,seq2) == sorted(seq3))

    print(merge_sorted_seqs(seq1,seq2)) # [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10]
    print('테스트통과')

if __name__ == '__main__':
    test_merge_sorted()


