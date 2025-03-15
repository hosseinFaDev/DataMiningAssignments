import pandas as pd
df = pd.read_csv('dist/adult_preprocessed.csv')

def find_frequent_itemsets(dataset: pd.DataFrame, min_support_count: int) -> list[set[str]]:
    # each node one itemset
    class FPTreeNode:
        def __init__(self, name, count, parent):
            self.name = name
            self.count = count
            self.parent = parent
            self.children = {}
            self.link = None

        def increment(self, count):
            self.count += count
    # header table for count each itemset
    def build_fptree(transactions, min_support):
        #header_table a dic witch contain frequent itemset data
        header_table = {}
        for transaction in transactions:
            for item in transaction:
                #get(item, 0)  => return item if is available ,if not return 0
                header_table[item] = header_table.get(item, 0) + 1

        # delete item with low minimum support with create new dic
        header_table = {k: v for k, v in header_table.items() if v >= min_support}
        if not header_table:
            return None, None

        # create links to first node in tree (key in dic change to new list(old value,none) )
        for k in header_table.keys():
            header_table[k] = [header_table[k], None]

        #create root node (name,repeat,parents)
        tree_root = FPTreeNode('null', 1, None)

        for transaction in transactions:
          # create and filter frequent_items list for each transaction (only item in header)
            frequent_items = [item for item in transaction if item in header_table]
            #sort them desk 
            frequent_items.sort(key=lambda x: header_table[x][0], reverse=True)

            if frequent_items:
                update_tree(frequent_items, tree_root, header_table)

        return tree_root, header_table

    #input(items list,current node , header_table(link))
    def update_tree(items, node, header_table):
        first_item = items[0]
        #check if first_item add before ,only increce one 
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
                                #(name,count,parent)
            new_node = FPTreeNode(first_item, 1, node)
            node.children[first_item] = new_node
            #add new node to header_table if we have't link before
            if header_table[first_item][1] is None:
                header_table[first_item][1] = new_node

            #if link exsist, add new link to last node
            else:
                current = header_table[first_item][1]
                while current.link is not None:
                    current = current.link
                current.link = new_node

        # if len(items)>1    => function execute recersively
        if len(items) > 1:
            update_tree(items[1:], node.children[first_item], header_table)

    ## extract frequent itemset 
    def mine_fptree(header_table, min_support, past_itemst, frequent_itemsets):
        sorted_items = sorted(header_table.items(), key=lambda x: x[1][0])

        #procesess each item form header_table
        for item, node_info in sorted_items:
            #copy from past_itemst
            new_past_itemst = past_itemst.copy()
            #add new frequent_itemsets
            new_past_itemst.add(item)
            frequent_itemsets.append(new_past_itemst)

            #conditional pattern PATH
            conditional_pattern_base = []
            #pointer to first node (in linked list)
            node = node_info[1]
            #navigation all node for specefic item
            while node is not None:
                # a list from PATH(from root to node)
                path = []
                parent = node.parent
                #navigation corrent node to root
                while parent.name != 'null':
                    #add parent to path
                    path.append(parent.name)
                    #navigation to parent of parent
                    parent = parent.parent
                # add PATH to conditional_pattern for each item repeate
                for _ in range(node.count):
                    conditional_pattern_base.append(path)
                #change pointer to next node in linked list
                node = node.link
            
            __, subtree_header = build_fptree(conditional_pattern_base, min_support)
            #check if subtree made before => extract frequent itemset 
            if subtree_header is not None:
                mine_fptree(subtree_header, min_support, new_past_itemst, frequent_itemsets)
    #change data format from pandas to transactions(need for algoritem) => by tolist()
    transactions = dataset.apply(lambda row: list(row.index[row == 1]), axis=1).tolist()
    tree, header_table = build_fptree(transactions, min_support_count)

    #if we don't have frequent itemset
    if not tree:
        return []

    frequent_itemsets = []
    mine_fptree(header_table, min_support_count, set(), frequent_itemsets)
    return frequent_itemsets


def generate_rules(frequent_itemsets: list[set[str]], min_confidence: float, dataset: pd.DataFrame) -> list[tuple[set[str], set[str]]]:

    # generate subsets from frequent_itemsets
    def generate_subsets(itemset):
        subsets = []
        # generate all subsets from frequent_itemsets
        def recursive_helper(items, subset):
            if items:
                recursive_helper(items[1:], subset + [items[0]])
                recursive_helper(items[1:], subset)
            elif subset:
                # means we are end of itemset list ( need to add to subset)
                subsets.append(set(subset))

        recursive_helper(list(itemset), [])
        return subsets

    # change to transactions => by tolist()
    transactions = dataset.apply(lambda row: set(row.index[row == 1]), axis=1).tolist()
    rules = []

    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            #create all subsets from itemset
            subsets = generate_subsets(itemset)
            for antecedent in subsets:
                # calculate consequent =>  {A, B, C} - {A, B} = {C}
                consequent = itemset - antecedent

                if consequent:
                    #calculate support & if Condition pass +1
                    antecedent_support = sum(1 for transaction in transactions if antecedent.issubset(transaction))
                    rule_support = sum(1 for transaction in transactions if itemset.issubset(transaction))
                    #calculate confidence
                    confidence = rule_support / antecedent_support if antecedent_support > 0 else 0
                    #check it has minimum confidence
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent))


    print("Rules with Support and Confidence Just Print:\n")
    #print Rules with Support and Confidence
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
           for antecedent in generate_subsets(itemset):
            consequent = itemset - antecedent
            if consequent:
                #calculate support & if Condition pass +1
                antecedent_support = sum(1 for t in transactions if antecedent.issubset(t))
                rule_support = sum(1 for t in transactions if itemset.issubset(t))
                confidence = rule_support / antecedent_support if antecedent_support > 0 else 0
                
                if confidence >= min_confidence:
                    support = rule_support / len(transactions)
                    antecedent_str = ', '.join(antecedent)
                    consequent_str = ', '.join(consequent)
                    print(f"({antecedent_str}) -> ({consequent_str}) | Support: {support:.4f}, Confidence: {confidence:.4f}")

 
    
    
    return rules


# set parameters
min_support_count = 13000
min_confidence = 0.95


frequent_itemsets = find_frequent_itemsets(df, min_support_count)

# save to text file
with open('dist/freq_itemsets.txt', 'w') as f:
    print('Frequent itemsets file generate sucsessfully')
    for itemset in frequent_itemsets:
        f.write(', '.join(itemset) + '\n')



rules = generate_rules(frequent_itemsets, min_confidence, df)

# save to text file
with open('dist/rules.txt', 'w') as f:
    print('Roles file generate sucsessfully')
    for antecedent, consequent in rules:
        f.write(f"({', '.join(antecedent)}) -> ({', '.join(consequent)})\n")
