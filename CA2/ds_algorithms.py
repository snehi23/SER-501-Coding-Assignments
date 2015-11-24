import doctest
__author__ = 'snehi23'


class SinglyLinkedNode(object):

    def __init__(self, item=None, next_link=None):
        super(SinglyLinkedNode, self).__init__()
        self._item = item
        self._next = next_link

    @property
    def item(self):
        return self._item

    @item.setter
    def item(self, item):
        self._item = item

    @property
    def next(self):
        return self._next

    @next.setter
    def next(self, next):
        self._next = next

    def __repr__(self):
        return repr(self.item)


class SinglyLinkedList(object):
    """
    >>> list = SinglyLinkedList()
    >>> list.append(5)
    >>> list.__len__()
    1
    >>> list = SinglyLinkedList()
    >>> list.append(4)
    >>> list.__contains__(4)
    True

    >>> list = SinglyLinkedList()
    >>> SinglyLinkedList().remove(5)
    'Item not present'

    >>> list = SinglyLinkedList()
    >>> list.append(5)
    >>> list.append(4)
    >>> list
    List:5->4

    >>> list = SinglyLinkedList()
    >>> list.append(5)
    >>> list.append(4)
    >>> list.remove(5)
    'Item removed'

    >>> list = SinglyLinkedList()
    >>> list.append(5)
    >>> list.append(4)
    >>> list.remove(6)
    'Item not present'

    """

    def __init__(self):
        super(SinglyLinkedList, self).__init__()
        self.head = None

    def __len__(self):
        current = self.head
        i = 0
        while current is not None:
            i += 1
            current = current.next
        print i

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.item
            node = node.next

    def __contains__(self, item):
        present = False
        current = self.head
        while current is not None:
            if current.item == item:
                present = True
            current = current.next
        return present

    def remove(self, item):
        if self.__contains__(item):
            current = self.head
            if current.item == item:
                self.head = current.next
            while current is not None and current.next is not None:
                if current.next.item == item:
                    current.next = current.next.next
                current = current.next
            return "Item removed"
        else:
            return "Item not present"

    def prepend(self, item):
        new_node = SinglyLinkedNode(item)
        new_node.next = self.head
        self.head = new_node

    def __repr__(self):
        s = "List:" + "->".join([str(item) for item in self])
        return s

    # ADDITIONAL FUNCTIONS TO ASSIST REQUIRED FUNCTIONS

    def append(self, item):
        current = self.head
        new_node = SinglyLinkedNode(item)
        if current is None:
            current = new_node
            self.head = current
        else:
            while current.next is not None:
                current = current.next
            current.next = new_node

    def show(self):
        node = self.head
        s = ""
        while node is not None:
            s += str(node.item) + " "
            node = node.next
        print s


class ChainedHashDict(object):
    """
    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> print dict
    {(1, 5)}

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__getitem__(1)
    5

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.__delitem__(1)
    >>> print dict
    {(2, 6)}

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__contains__(1)
    True

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.__len__()
    2

    >>> dict = ChainedHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.display()
    Bin0 : Empty
    Bin1 : Empty
    Bin2 : Empty
    Bin3 :  (1, 5)
    Bin4 :  (2, 6)
    Bin5 : Empty
    Bin6 : Empty
    Bin7 : Empty
    Bin8 : Empty
    Bin9 : Empty

    """
    def __init__(self, bin_count=10, max_load=0.3, hashfunc=hash):
        super(ChainedHashDict, self).__init__()
        self.table = [[] for x in range(bin_count)]
        self._bin_count = bin_count
        self._load_factor = max_load

    @property
    def load_factor(self):
        return self._load_factor

    @property
    def bin_count(self):
        return self._bin_count

    def rebuild(self, bin_count):
        self._bin_count = bin_count
        temp_list = []
        for bucket in self.table:
            if len(bucket) != 0:
                temp_list.extend(bucket)
        self.table = [[] for x in range(bin_count)]
        for (k, v) in temp_list:
            self.__setitem__(k, v)

    def __getitem__(self, key):
        list = self.__getIndex__(key)
        for (k, v) in list:
            if k is key:
                return v
        return None

    def __setitem__(self, key, value):
        if float(self.__len__()) / self.bin_count >= self.load_factor:
            self.rebuild(self.bin_count + 5)
        list = self.__getIndex__(key)
        for (k, v) in list:
            if k is not key:
                list.append((key, value))
            else:
                list[list.index((k, v))] = (key, value)
            return
        list.append((key, value))

    def __delitem__(self, key):
        list = self.__getIndex__(key)
        for (k, v) in list:
            if k is key:
                list.remove((k, v))

    def __contains__(self, key):
        list = self.__getIndex__(key)
        for (k, v) in list:
            if k is key:
                return True
        return False

    def __len__(self):
        count = 0
        for list in self.table:
                count += len(list)
        return count

    def display(self):
        i = 0
        while i < len(self.table):
            if self.table[i]:
                result = ''
                print 'Bin' + str(i) + ' : ',
                list = self.table[i]
                for (k, v) in list:
                    result = result + str((k, v)) + ','
                print result[:-1]
                del list[:]
            else:
                print 'Bin' + str(i) + ' : ' + 'Empty'
            i += 1

    # ADDITIONAL FUNCTIONS TO ASSIST REQUIRED FUNCTIONS

    def __hash__(self, key):
        return (key + 2) % self.bin_count

    def __getIndex__(self, key):
        return self.table[self.__hash__(key)]

    def __str__(self):
        result = '{'
        for list in self.table:
            for (k, v) in list:
                result = result + str((k, v)) + ','
        return result[:-1] + '}'


class OpenAddressHashDict(object):

    """
    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> print dict
    {(1, 5)}

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__getitem__(1)
    5

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.__delitem__(1)
    >>> print dict
    {('DEL', 'DEL'),(2, 6)}

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__contains__(1)
    True

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.__len__()
    2

    >>> dict = OpenAddressHashDict()
    >>> dict.__setitem__(1,5)
    >>> dict.__setitem__(2,6)
    >>> dict.display()
    Bin0 : Empty
    Bin1 : Empty
    Bin2 : Empty
    Bin3 :  (1, 5)
    Bin4 :  (2, 6)
    Bin5 : Empty
    Bin6 : Empty
    Bin7 : Empty
    Bin8 : Empty
    Bin9 : Empty

    """

    def __init__(self, bin_count=10, max_load=0.3, hashfunc=hash):
        super(OpenAddressHashDict, self).__init__()
        self.table = [None] * bin_count
        self._bin_count = bin_count
        self._load_factor = max_load

    @property
    def load_factor(self):
        return self._load_factor

    @property
    def bin_count(self):
        return self._bin_count

    def rebuild(self, bin_count):
        self._bin_count = bin_count
        temp_list = []
        temp_list.extend([element for element in self.table if element is not None])
        self.table = [None] * bin_count
        for (k, v) in temp_list:
           self.__setitem__(k,v)

    def __getitem__(self, key):
        i = 0
        while i < len(self.table):
            j = self.__getIndex__(key, i)
            if self.table[j] and self.table[j][0] == key:
                return self.table[j][1]
            else:
                pass
            i += 1
        return "Not found"

    def __setitem__(self, key, value):
        i = 0
        if float(self.__len__()) / self.bin_count >= self.load_factor:
            self.rebuild(self.bin_count + 5)
        else:
            while i < len(self.table):
                j = self.__getIndex__(key, i)
                if self.table[j] and (self.table[j][0] == key or self.table[j][0] == 'DEL'):
                    self.table[j] = (key, value)
                    return
                elif self.table[j] and self.table[j][0] != key:
                    pass
                else:
                    self.table[j] = (key, value)
                    return
                i += 1

    def __delitem__(self, key):
        i = 0
        while i < len(self.table):
            j = self.__getIndex__(key, i)
            if self.table[j] and self.table[j][0] == key:
                self.table[j] = ('DEL', 'DEL')
            else:
                pass
            i += 1

    def __contains__(self, key):
        i = 0
        while i < len(self.table):
            j = self.__getIndex__(key, i)
            if self.table[j] and self.table[j][0] == key:
                return True
            else:
                pass
            i += 1
        return False

    def __len__(self):
        count = 0
        for element in self.table:
            if element is not None and element is not ('DEL', 'DEL'):
                count += 1
        return count

    def display(self):
        i = 0
        while i < len(self.table):
            if self.table[i]:
                result = ''
                print 'Bin' + str(i) + ' : ',
                print self.table[i]
                result = result + str((self.table[i][0], self.table[i][1])) + ','
            else:
                print 'Bin' + str(i) + ' : ' + 'Empty'
            i += 1

    # ADDITIONAL FUNCTIONS TO ASSIST REQUIRED FUNCTIONS

    def __hash__(self, key):
        return (key + 2) % self.bin_count

    def __linear_prob_hash__(self, key, i):
        return (self.__hash__(key) + i) % self.bin_count

    def __getIndex__(self, key, i):
        return self.__linear_prob_hash__(key, i)

    def __str__(self):
        result = '{'
        for pair in self.table:
                if pair is not None:
                    result = result + str((pair[0], pair[1])) + ','
        return result[:-1] + '}'


class BinaryTreeNode(object):
    def __init__(self, key, val, parent, left=None, right=None):
        super(BinaryTreeNode, self).__init__()
        self.key = key
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return '(' + str(self.key) + ', ' + str(self.val) + ')'


class BinarySearchTreeDict(object):

    """
    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.height()
    3

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.inorder_keys()
    3 4 6 7 8

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.postorder_keys()
    3 4 7 8 6

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.preorder_keys()
    6 4 3 8 7

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.items()
    (3, 40)(4, 20)(6, 10)(7, 60)(8, 30)

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.__getitem__(6)
    10

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.__setitem__(6,20)
    >>> tree.items()
    (3, 40)(4, 20)(6, 20)(7, 60)(8, 30)

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.__delitem__(7)
    >>> tree.items()
    (3, 40)(4, 20)(6, 10)(8, 30)

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.__contains__(7)
    True

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.display()
    3 4 6 7 8
    6 4 3 8 7

    >>> tree = BinarySearchTreeDict()
    >>> tree.insert(6,10)
    >>> tree.insert(4,20)
    >>> tree.insert(8,30)
    >>> tree.insert(3,40)
    >>> tree.insert(7,60)
    >>> tree.height()
    3

    """

    def __init__(self):
        super(BinarySearchTreeDict, self).__init__()
        self.root = None

    def height(self):
        print ""+str(self.height_of_tree(self.root))

    def inorder_keys(self):
        print " ".join([str(node) for node in self.inorder_traverse(self.root)])

    def postorder_keys(self):
        print " ".join([str(node) for node in self.postorder_traverse(self.root)])

    def preorder_keys(self):
        print " ".join([str(node) for node in self.preorder_traverse(self.root)])

    def items(self):
        self.getitems(self.root)

    def __getitem__(self, key):
        self.getitem(self.root, key)

    def __setitem__(self, key, value):
        self.setitem(self.root, key, value)

    def __delitem__(self, key):
        self.delitem(self.root, key)

    def __contains__(self, key):
        self.contains(self.root, key)

    def __len__(self):
        self.len(self.root)

    def display(self):
        self.inorder_keys()
        self.preorder_keys()

    def height_of_tree(self, node):
        if node is None:
            return 0
        else:
            return 1 + max(self.height_of_tree(node.left), self.height_of_tree(node.right))

    # ADDITIONAL FUNCTIONS TO ASSIST REQUIRED FUNCTIONS

    def findNode(self, node, key):
        if node is None:
            print "Key not found"
        elif node.key is key:
            return node
        elif key < node.key:
            return self.findNode(node.left, key)
        else:
            return self.findNode(node.right, key)

    def insert(self, key, val):
        if self.root is None:
            self.root = BinaryTreeNode(key, val, None)
        else:
            current = self.root
            while current:
                if key <= current.key:
                    if current.left:
                        current = current.left
                    else:
                        current.left = BinaryTreeNode(key, val, current.key)
                        break
                else:
                    if current.right:
                        current = current.right
                    else:
                        current.right = BinaryTreeNode(key, val, current.key)
                        break

    def inorder_traverse(self, node):

        if node.left is not None:
            for child_node in self.inorder_traverse(node.left):
                yield child_node
        yield node
        if node.right is not None:
            for child_node in self.inorder_traverse(node.right):
                yield child_node


    def postorder_traverse(self, node):

        if node.left is not None:
            for child_node in self.postorder_traverse(node.left):
                yield child_node
        if node.right is not None:
            for child_node in self.inorder_traverse(node.right):
                yield child_node
        yield node

    def preorder_traverse(self, node):

        yield node
        if node.left is not None:
            for child_node in self.preorder_traverse(node.left):
                yield child_node
        if node.right is not None:
            for child_node in self.preorder_traverse(node.right):
                yield child_node

    def getitem(self, node, key):
        if node is None:
            print "Key not found"
        elif node.key is key:
            print node.val
        elif key < node.key:
            return self.getitem(node.left, key)
        else:
            return self.getitem(node.right, key)

    def setitem(self, node, key, value):
       if node is None:
           pass
       elif node.key is key:
           node.val = value
       elif key < node.key:
           return self.setitem(node.left, key, value)
       else:
           return self.setitem(node.right, key, value)

    def delitem(self, node, key):
        node_to_delete = self.findNode(node, key)
        parent = self.findNode(node, node_to_delete.parent)

        if self.children_count(node_to_delete) is 0:
            if parent:
                if parent.left is node_to_delete:
                    parent.left = None
                else:
                    parent.right = None
                del node_to_delete

        elif self.children_count(node_to_delete) is 1:
            if node_to_delete.left:
                child_node = node_to_delete.left
            else:
                child_node = node_to_delete.right
            if parent:
                if parent.left is node_to_delete:
                    parent.left = child_node
                else:
                    parent.right = child_node
                del node_to_delete
        else:
            parent = node_to_delete
            successor = node_to_delete.right
            while successor.left:
                parent = successor
                successor = successor.left
            node_to_delete.key = successor.key
            node_to_delete.val = successor.val
            if parent.left is successor:
                parent.left = successor.right
            else:
                parent.right = successor.right

    def getitems(self, node):
        print "".join([repr(node) for node in self.inorder_traverse(self.root)])

    def children_count(self, node):
        count = 0
        if node.left:
            count += 1
        if node.right:
            count += 1
        return count

    def contains(self, node, key):
        if node is None:
            print False
        elif node.key is key:
            print True
        elif key < node.key:
            return self.contains(node.left, key)
        else:
            return self.contains(node.right, key)

    def len(self, node):
        if node is None:
            return 0
        else:
            return 1 + self.len(node.left) + self.len(node.right)


def terrible_hash(bin):
    """A terrible hash function that can be used for testing.

    A hash function should produce unpredictable results,
    but it is useful to see what happens to a hash table when
    you use the worst-possible hash function.  The function
    returned from this factory function will always return
    the same number, regardless of the key.

    :param bin:
        The result of the hash function, regardless of which
        item is used.

    :return:
        A python function that can be passes into the constructor
        of a hash table to use for hashing objects.
    """
    def hashfunc(item):
        return bin
    return hashfunc


def main():

    doctest.testmod()

if __name__ == '__main__':
    main()
