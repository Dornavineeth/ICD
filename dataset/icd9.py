import csv
import json
from collections import *


class Node(object):
    def __init__(self, depth, code, descr=None):
        self.depth = depth
        self.descr = descr or code
        self.code = code
        self.parent = None
        self.children = []
        self.subtree_depth_ = None

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
        self.children = sorted(self.children, key=lambda x: x.code)

    def search(self, code):
        if code == self.code:
            return [self]
        ret = []
        for child in self.children:
            ret.extend(child.search(code))
        return ret

    def find(self, code):
        nodes = self.search(code)
        if nodes:
            return nodes[0]
        return None

    def subtree_depth(self):
        if len(self.children) == 0:
            return 1
        if self.subtree_depth_ is not None:
            return self.subtree_depth_
        c = 0
        for child in self.children:
            c = max(c, child.subtree_depth())
        self.subtree_depth_ = c+1
        return c + 1

    @property
    def root(self):
        return self.parents[0]

    @property
    def description(self):
        return self.descr

    @property
    def codes(self):
        return map(lambda n: n.code, self.leaves)

    @property
    def parents(self):
        n = self
        ret = []
        while n:
            ret.append(n)
            n = n.parent
        ret.reverse()
        return ret

    @property
    def leaves(self):
        leaves = set()
        if not self.children:
            return [self]
        for child in self.children:
            leaves.update(child.leaves)
        return list(leaves)


    # return all leaf notes with a depth of @depth
    def leaves_at_depth(self, depth):
        return filter(lambda n: n.depth == depth, self.leaves)

    @property
    def siblings(self):
        parent = self.parent
        if not parent:
            return []
        return list(parent.children)

    def __str__(self):
        return "%s\t%s" % (self.depth, self.code)

    def __hash__(self):
        return hash(str(self))


class ICD9(Node):
    def __init__(self, codesfname=None, semantic_sep="-", allcodes=None):
        # dictionary of depth -> dictionary of code->node
        self.depth2nodes = defaultdict(dict)
        self.semantic_sep = semantic_sep
        super(ICD9, self).__init__(-1, "ROOT")

        if allcodes is None:
            with open(codesfname, "r") as f:
                allcodes = json.loads(f.read())
        self.process(allcodes)

    def process(self, allcodes):
        for hierarchy in allcodes:
            self.add(hierarchy)

    def get_node(self, depth, code, descr):
        d = self.depth2nodes[depth]
        if code not in d:
            d[code] = Node(depth, code, descr)
        return d[code]

    def _create_semantic_id(self, node):
        prefix = node.semantic_id+self.semantic_sep
        prefix = prefix.lstrip(self.semantic_sep)
        for idx, c in enumerate(node.children):
            node.children[idx].semantic_id = prefix+str(idx)
            self._create_semantic_id(node.children[idx])
    
    def _get_total_num_nodes(self, node):
        n = 1
        for c in node.children:
            n += self._get_total_num_nodes(c)
        return n
    
    def _get_max_children(self, node):
        c = len(node.children)
        for child in node.children:
            c = max(c, self._get_max_children(child))
        return c

    def max_children(self):
        c = 0
        return max([self._get_max_children(x) for x in self.children])

    def get_total_nodes(self):
        n = self._get_total_num_nodes(node=self)
        return n-1

    def create_semantic_id(self):
        self.semantic_id = ''
        self._create_semantic_id(self)


    def get_semantic_id(self, key):
        node = self.find(key)
        if node is None:
            return None
        return node.semantic_id

    def max_children_level(self, level):
        if level>=self.subtree_depth():
            return None
        if self._max_children_level is None:
            self._max_children_level = self._init_max_children_level()
        return self._max_children_level.get(level, None)

    def _init_max_children_level(self):
        self._max_children_level = {}
        d = self.subtree_depth()
        curr = [self]
        for d in range(d-1):
            max_children = 0
            next_curr = []
            for c in curr:
                max_children = max(max_children, len(c.children))
                next_curr += list(c.children)
            curr = next_curr
            self._max_children_level[d] = max_children


    def add(self, hierarchy):
        prev_node = self
        for depth, link in enumerate(hierarchy):
            if not link["code"]:
                continue

            code = link["code"]
            descr = "descr" in link and link["descr"] or code
            node = self.get_node(depth, code, descr)
            node.parent = prev_node
            prev_node.add_child(node)
            prev_node = node


if __name__ == "__main__":
    # tree = ICD9("codes.json")
    tree = ICD9('codes_icd9_diagnosis_2015.json')
    counter = Counter(map(str, tree.leaves))
    print(tree.subtree_depth())
    print(tree.max_children())
    print(tree.get_total_nodes())
    tree.create_semantic_id()
    tree._init_max_children_level()

    import pickle

    tree._init_max_children_level()
    tree._max_children_level

    with open('mimic_train_tree.pkl', 'wb') as f:
        pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)

    with open('mimic_train_tree.pkl', 'rb') as f:
        tree = pickle.load(f)

    print(tree._max_children_level)