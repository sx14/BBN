class Node:
    def __init__(self, id, raw_id=None):
        self.id = id
        self.raw_id = raw_id
        self.children = []
        self.parent = None

    def get_level(self):
        return self.id.split('_')[0]

    def add_child(self, node):
        self.children.append(node)

    def is_raw(self):
        return self.raw_id is not None

    def is_leaf(self):
        return len(self.children) == 0

    def get_leaf_ids(self):
        leaf_ids = []

        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            if curr.is_leaf():
                leaf_ids.append(curr.raw_id)
            else:
                for child in curr.children:
                    queue.append(child)
        return leaf_ids

    def get_leaves(self):
        leaves = []

        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            if curr.is_leaf():
                leaves.append(curr)
            else:
                for child in curr.children:
                    queue.append(child)
        return leaves

    def get_raws(self, level):
        raws = []

        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            # if curr.is_raw() and curr != self:
            if curr.is_raw() and curr.get_level() == level:
                raws.append(curr)
            else:
                for child in curr.children:
                    queue.append(child)
        return raws

    def get_raw_ids(self, level):
        raw_ids = []

        queue = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            # if curr.is_raw() and curr != self:
            if curr.is_raw() and curr.get_level() == level:
                raw_ids.append(curr.raw_id)
            else:
                for child in curr.children:
                    queue.append(child)
        return raw_ids



class Tree:

    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        if node.id in self.nodes:
            raise ValueError('Node[%s] exists!' % node.id)
        self.nodes[node.id] = node

    def add_edge(self, parent_id, child_id):
        pNode = self.nodes[parent_id]
        cNode = self.nodes[child_id]
        pNode.add_child(cNode)
        cNode.parent = pNode

    def size(self):
        return len(self.nodes)

    def get(self, id):
        return self.nodes[id]

    def get_root(self):
        root = None
        for id, node in self.nodes.items():
            if node.parent is None:
                assert root is None
                root = node
        return root

    def get_leaves(self):
        root = self.get_root()
        return root.get_leaves()

    def get_raws(self, level):
        root = self.get_root()
        queue = [root]
        raws = []
        while len(queue) > 0:
            curr = queue.pop(0)
            if curr.is_raw() and level in curr.id:
                raws.append(curr)
            else:
                for child in curr.children:
                    queue.append(child)
        return raws

    def get_non_leaf_nodes(self):
        return {id: node for id, node in self.nodes.items() if not node.is_leaf()}

