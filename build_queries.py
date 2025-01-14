# -*- coding: utf-8 -*-
"""
Updated last October 3, 2024

@author: oxf7
"""
from collections import defaultdict
from pathlib import Path
import re

import pandas as pd

class Query():
    
    __version__ = "4.0"
    
    def __init__(self, queries, query_type="boundary", input_type="csv"):
        
        self.input_type = input_type
        try:
            assert self.input_type in ["csv", "excel", "list", "dataframe"]
        except AssertionError:
            raise ValueError("The 'input_type' keyword must be one of the following: 'csv', 'excel', 'list', 'dataframe'. Default is 'csv'")
        

        allowable_query_types = ["join", "join with boundary", "no boundary", "boundary", "boundary with s"]
        if query_type in allowable_query_types:
            self.query_type = query_type
        else:
            raise ValueError(f"Unrecognized query type {query_type}, use one of {allowable_query_types}")
            
        if self.input_type =="list":
            self.term_df = pd.DataFrame({"TERM": queries})
            self.name = "Custom List"
            
        elif self.input_type == "excel":
            self.term_df = pd.read_excel(Path(queries), converters = {0:str})
            self.name = "Excel"            

        elif self.input_type == "csv":
            self.term_df = pd.read_csv(Path(queries), converters = {0:str})
            self.name = "CSV"
            
        else:
            self.name = "Pandas dataframe"
            self.term_df = queries.copy()

        if self.term_df.shape[1] ==1:
            self.term_df.columns = ['TERM'] + [x for x in self.term_df.columns[1:]]
        else:
            self.term_df.columns = ['TERM', 'CATEGORY'] + [x for x in self.term_df.columns[2:]]
            if self.term_df[self.term_df['CATEGORY'].isna()].shape[0] != 0:
                raise ValueError("If you pass in a file with both terms and categories, all terms must have a category")
    


    def build_re(self):

        lines = [x.strip().lower() for x in self.term_df['TERM'].unique()] 
             
        self.expression_set = set()
        mixed_type = False
        for line in lines:
#            line = line.split(",")[0] #sometimes single term per line, sometimes 1st term in csv, to do re.split(r'[\t,]')
#            line = re.split(r"[,\t]", line)[0]
            if line.endswith("*"):
                mixed_type = True
            self.expression_set.add(line.lower().strip())
                    
        self.expression_list = sorted(self.expression_set, key=lambda x: len(x), reverse=True)
        if self.query_type == 'join':
            self.re = self.join_re()
        elif self.query_type == 'join with boundary':
            self.re = self.join_wb_re()
        elif mixed_type:
            self.query_type = "join_wb_re"
            self.re = self.join_wb_re()
        else:
            self.re = self.trie_re()
        return(self.re)
    
    def join_re(self): 

        catREs = sorted([r'%s' % x for x in self.expression_list])
        catREString= "|".join(catREs)
        print(f"Successfully built a regex join for {self.name} with {len(catREs)} items")    
        return(re.compile(catREString, flags=re.IGNORECASE))

    def join_wb_re(self): 
        #TODO: make single boundary in regex at beg and end, and test function
        catREs = []
        for x in self.expression_list:
            if "*" in x:
                x = x.replace('*', '')
                catREs.append(r'\b%s' % x)
            else:
                catREs.append(r'\b%s\b' % x)
#        catREs = sorted([r'\b%s\b' % x for x in self.expression_list])
        catREString= "|".join(catREs)
        print(f"Successfully built a regex join for {self.name} with {len(catREs)} items")    
        return(re.compile(catREString, flags=re.IGNORECASE))
          
    def trie_re(self):
        trie = Trie()
        for searchTerm in self.expression_list:
            trie.add(searchTerm)
        if self.query_type=="no boundary":
            searchString = r'%s' % trie.pattern()
            print(f"Successfully built trie regex query, no word boundaries, for {len(self.expression_list)} items") 
        elif self.query_type=="boundary with s":
            searchString = r'\b%ss?\b' % trie.pattern()
            print(f"Successfully built trie regex query, adding word boundaries, optional s for file:{self.name} for {len(self.expression_list)} items")
        else: #boundary condition
            searchString = r'\b%s\b' % trie.pattern()
            print(f"Successfully built trie regex query, adding word boundaries, for file:{self.name} for {len(self.expression_list)} items")     
        
        return(re.compile(searchString, flags=re.IGNORECASE))

    
    def build_category_map(self ):
        try:
            assert self.term_df.shape[1] > 1
        except AssertionError:
            raise AssertionError("Building a category map requires a 2-column file input rather than a list")            
        
        lookup_d = defaultdict(set)
        for _, row in self.term_df.iterrows():
            term = row['TERM'].strip().lower()
            category = row['CATEGORY'].strip().upper()
            lookup_d[term].add(category)
            if not term.endswith('s'):
                lookup_d[term + "s"].add(category)

        return(lookup_d)
            

class Trie():
    """Regex::Trie in Python. Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union. Found in stackoverflow, where was link?"""

    def __init__(self):
        self.data = {}

    def add(self, word):
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[''] = 1

    def dump(self):
        return self.data

    def quote(self, char):
        return re.escape(char)

    def _pattern(self, pData):
        data = pData
        if "" in data and len(data.keys()) == 1:
            return None

        alt = []
        cc = []
        q = 0
        for char in sorted(data.keys()):
            if isinstance(data[char], dict):
                try:
                    recurse = self._pattern(data[char])
                    alt.append(self.quote(char) + recurse)
                except:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = "(?:%s)?" % result
        return result

    def pattern(self):
        return self._pattern(self.dump())
