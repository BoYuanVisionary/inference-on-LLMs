from reasoning.evaluator.math_grader import extract_answer

class treeNode(object):
    def __init__(self, pcd, parent=None, depth=0):
        self.pcd = pcd  # str
        self.y = ''  # str the whole partial solution up tp now
        self.parent = parent  # treeNode
        self.numVisits = 0  # int
        self.V = 0  # float
        self.children = {}  # dict{str:treeNode}
        self.depth = depth  # int
        self.isFullyExpanded = False  # expanded
        self.visit_sequence = 0
        self.final_ans_flag = 0
        self.reflection = ''
        self.isTerminal = False  # value acceptable
        self.on_final_route = False
        self.min_steps_to_correct = 1024
        self.summary = ''
        self.he = 0  # hard estimation
        self.se = 0  # soft estimation

    def __str__(self):
        s = ["numVisits: %d" % self.numVisits, 
             f'V:{self.V}', "possibleActions: %s" % (self.children.keys()), 
             f'he:{self.he}', f'se:{self.se}', 
             f'isFullyExpanded:{self.isFullyExpanded}', 
             f'visit_sequence:{self.visit_sequence}', 
             f'final_ans_flag:{self.final_ans_flag}', 
             f'isTerminal:{self.isTerminal}', 
             f'on_final_route:{self.on_final_route}',
             f'pcd:{self.pcd}']
        return "%s: {%s}" % (self.__class__.__name__, ', '.join(s))

    def append_children(self, new_pcd: str):
        node = treeNode(new_pcd, self, self.depth + 1)
        node.update_y_from_parent()
        self.children.update({new_pcd: node})
        return node

    def update_y_from_parent(self):
        if self.parent is None:
            self.y = self.pcd
        else:
            self.y = self.parent.y + self.pcd
        self.update_is_terminal()
    
    def update_is_terminal(self): # currently only for math problem
        if extract_answer(self.y) is not None or '\boxed' in self.y:
            self.isTerminal = True
        else:
            self.isTerminal = False

    def getBestV(self):  # get the best node, don't have to be leaf node
        
        if len(self.children) == 0:
            return self, self.V
        max_V = self.V
        max_node = self
        for child in self.children.values():
            subNode, subValue = child.getBestV()
            if subValue > max_V:
                max_V = subValue
                max_node = subNode
        return max_node, max_V
    
    def getBestTerminalV(self):
        
        if self.isTerminal:
            current_max = self.V
            current_node = self
        else:
            current_max = -float('inf')
            current_node = None

        for child in self.children.values():
            child_node, child_value = child.getBestTerminalV()
            if child_value > current_max:
                current_max = child_value
                current_node = child_node
                
        if current_node is None and self.isTerminal:
            return self, self.V
            
        return current_node, current_max

    def trace_route(self):  # trace route from terminal node to root
        cur_node = self
        while cur_node is not None:
            cur_node.on_final_route = True
            cur_node = cur_node.parent
            
    def print_tree(self, indent=0, last=True, max_depth=None, show_fields=None):
        """
        Visualize the tree structure
        Parameters:
            indent: Current indentation level (starting from 0)
            last: Whether it is the last child node of the parent node
            max_depth: Maximum display depth
            show_fields: List of fields to display
        """
        if max_depth is not None and self.depth > max_depth:
            return
            
        default_fields = ['pcd', 'V', 'numVisits', 'isTerminal', 'depth']
        fields = show_fields or default_fields
        
        info_parts = []
        for field in fields:
            value = getattr(self, field, 'N/A')
            if isinstance(value, float):
                value = f"{value:.2f}"
            info_parts.append(f"{field}={value}")
        node_info = ", ".join(info_parts)
        
        if indent == 0:  
            prefix = ""
        else:
            prefix = "    " * (indent-1)  
            if last:
                prefix += "└── "
            else:
                prefix += "├── "

        print(f"{prefix}{node_info}")
        
        count = len(self.children)
        for i, (action, child) in enumerate(self.children.items()):
            is_last = i == count - 1
            child.print_tree(indent + 1, is_last, max_depth, show_fields)

