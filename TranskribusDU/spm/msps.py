# -*- coding: utf-8 -*-
"""

    cpy Xerox 2016
    
    Hervé Déjean
    XRCE
    READ project     

    initial code found atn https://github.com/dinsaurabh123/msps
    
"""

import math  
import itertools
from collections import Counter

class msps(object):
   
    def __init__(self): 
        self.output_patterns=[]
        self.sdc=None

        self.actual_supports = None
        self.lFeatures= None
        self.lMIS= None
        self.bDEBUG = False
  
    def setSDC(self,s):self.sdc=s
    def setFeatures(self,lf):
        self.lFeatures= lf
    def setMIS(self,l): self.lMIS= l

    def begin_msps(self,sequences):
      if (  sequences == None or len(sequences) == 0 or
            self.lMIS == None or len(self.lMIS) == 0  ):
        print 'Invalid data sequence or minimum support values'
        return None;
      
      # Total no. of data sequences
      sequence_count = len(sequences)
      output_patterns = []
      # Get the item support for each item i.e. sup(i)
      flattened_sequences = [ list(set(itertools.chain(*sequence))) for sequence in sequences ]
      if self.bDEBUG:print flattened_sequences
      if self.bDEBUG: print "MIS:",self.lMIS
      support_counts = dict(Counter(item for flattened_sequence in flattened_sequences for item in flattened_sequence))
      self.actual_supports = {item:support_counts.get(item)/float(sequence_count) for item in support_counts.keys()}
#       if self.bDEBUG:
      print "actual supports: %s" % self.actual_supports
      del flattened_sequences
      
      # Get the sorted list of frequent items i.e items with sup(i) >= MIS(i)
      frequent_items = sorted([item for item in self.actual_supports.keys() if self.actual_supports.get(item) >= self.lMIS.get(item)],key=self.lMIS.get)

      if self.bDEBUG:print "FrequentItems:",frequent_items
      
      # Iterate through frequent items to get sequential patterns
      for item in frequent_items:
        # Get the minimum item support count for item i.e count(MIS(item))
#         print item, self.lMIS.get(item), self.actual_supports.get(item)
        try:        
            mis_count = int(math.ceil(self.lMIS.get(item)*sequence_count))
        except TypeError:
            # issue in featureGeneration: to be fixed!!
            mis_count = 0.00
        
        if self.bDEBUG:print "------------- Current item:",item,"MIS:",mis_count, "Sup:",support_counts.get(item),"-----------------"
        if self.bDEBUG:print "Seq:", [sequence for sequence in sequences if self.has_item(sequence, item)]
           
        # Get the sequences containing that item and filter them to remove elements that do not satisfy SDC i.e. |sup(j) - sup(item)| > sdc
        item_sequences = [self.sdc_filter_on_item(sequence, item, self.actual_supports.get(item), self.actual_supports, self.sdc) for sequence in sequences if self.has_item(sequence, item)]
        if self.bDEBUG:print "ItemSeq:"
        if self.bDEBUG:print "\n".join([str(sequence) for sequence in item_sequences])
        
        
        # Run the restricted Prefix-Span to get sequential patterns
        self.r_prefix_span(output_patterns,item, item_sequences, mis_count)
        
        # Remove the item from original sequences
        sequences = self.remove_item(sequences, item)
        
      # End of the mining algorithm, print output
      if self.bDEBUG:self.write_output(output_patterns)
      return output_patterns
    
    def write_output(self,output_list):
      output_list = sorted(output_list,key=self.pattern_length)    # Sort the output based on pattern length
      output_text = ''    # Intialize empty string to append the output text
      
      cur_length = 1    # Initialize current length as 1 
      
      while True:   # Iterate until there are no patterns of specified length
        # Get all the patterns of current length from output 
        cur_length_patterns = filter (lambda a: self.pattern_length(a) == cur_length, output_list)
        if not cur_length_patterns:   # Break from the loop if the list is empty 
          break
        
        # Print the current length and number of patterns for current length
        output_text += "The number of length " + str(cur_length) + " sequential patterns is " + str(len(cur_length_patterns)) + "\n"
        
        # Print all the patterns with their support counts
        for (pattern,sup_count) in cur_length_patterns:
          str_pattern = "<{" + "}{".join([",".join(map(lambda x:str(x),itemset)) for itemset in pattern]) + "}>"
          output_text += "Pattern: " + str_pattern + " Count: " + str(sup_count) + "\n"
        
        cur_length += 1   # Increment the current length 
        
        output_text += "\n"
      
      print output_text
    #   output_file = open(out_file, 'w')
    #   output_file.write(output_text)
    #   output_file.close()
        
          
    def pattern_length(self,output_tuple):
      seq_pattern = output_tuple[0]   # Get the sequential pattern
      
      while isinstance(seq_pattern[0], list):   # Chain it until all sub-lists are removed
        seq_pattern = list(itertools.chain(*seq_pattern))
      
      return len(seq_pattern)   # Return the length of the pattern
    
    
    def remove_item(self,source_list, item_to_del):
      filtered_list = []    # Declare list to contain filter results
      # Check to see if list has sub lists as items
         
      if source_list and isinstance(source_list[0], list):
        for child_list in source_list:
          filtered_child_list = self.remove_item(child_list, item_to_del)    # Recurse for filtering each child_list
          
          if filtered_child_list:   # Append only non-empty lists
            filtered_list.append(filtered_child_list)  
        
      else:   # Remove item from the list
        filtered_list = filter (lambda a: a != item_to_del, source_list)
#         tmp=[]
#         for i in source_list:
#             print i, item_to_del,i != item_to_del,i == item_to_del
#             if i != item_to_del: tmp.append(i)
      return filtered_list    # Return the filtered list from current recursion
      
      
    def sdc_filter_on_item(self,source_list, base_item, base_item_support, supports, sd_constraint):
      filtered_list = []    # Declare list to contain filter results
      
      # Check to see if list has sub lists as items    
      if source_list and isinstance(source_list[0], list):
        for child_list in source_list:
          filtered_child_list = self.sdc_filter_on_item(child_list, base_item, base_item_support, supports, sd_constraint)    # Recurse for filtering each child_list
          if filtered_child_list:   # Append only the non-empty lists
            filtered_list.append(filtered_child_list)  
        
      else:   # Remove items that do not satisfy support difference constraint
        for item in source_list:
    #      print "Item:",item,"Support:",item_supports.get(item),"Diff:",abs(item_supports.get(item) - base_item_support)
          if not item == base_item and abs(supports.get(item) - base_item_support) > sd_constraint:  # Item doesn't satisfy SDC
            continue
          else:   # Item satisfies SDC
            filtered_list.append(item)
    
      return filtered_list    # Return the filtered list from current recursion
    
    
    def r_prefix_span(self,output_patterns,base_item, item_sequences, mis_count):
      # Get the frequent items and construct length one frequent sequences from them
      freq_item_sequences = self.remove_infrequent_items(item_sequences, mis_count)
      frequent_items = list(set(itertools.chain(*(itertools.chain(*freq_item_sequences)))))
      del freq_item_sequences
      
      # Get length-1 frequent sequences
      len_1_freq_sequences = [ [[item]] for item in frequent_items ]
    #  print len_1_freq_sequences
      
      # Remove the infrequent items
      item_sequences = self.remove_infrequent_items(item_sequences, mis_count)
      
      # Add the base_item 1-length sequential pattern to the output database
      if self.has_item(len_1_freq_sequences, base_item):
        output_patterns.append(([[base_item]], self.support_count(item_sequences, base_item)))
      
      # Run Prefix Span for each length-1 frequent sequential pattern    
      for freq_sequence in len_1_freq_sequences:
        self.prefix_span(output_patterns,freq_sequence, item_sequences, base_item, mis_count)
      
    
    def prefix_span(self,output_patterns,prefix, item_sequences, base_item, mis_count):
      if self.bDEBUG:print "Prefix:",prefix
        
      # Compute the projected database for the current prefix
      projected_db = self.compute_projected_database(prefix, item_sequences, base_item, mis_count)
      if self.bDEBUG:print "DB:"
      if self.bDEBUG:print "\n".join([str(sequence) for sequence in projected_db])
      
      # Find the prefix_length + 1 sequential patterns 
      if projected_db:    # Check if the projected database has any sequences
        
        # Initialize local variables
        prefix_last_itemset = prefix[-1]    # Last itemset in prefix
        all_template_1_items = []   # To hold all items for template 1 match i.e {30, x} or {_, x}
        all_template_2_items = []   # To hold all items for template 2 match i.e {30}{x}
        
        for proj_sequence in projected_db:
          itemset_index = 0
          template_1_items = []   # To hold items for template 1 match from current sequence
          template_2_items = []   # To hold items for template 2 match from current sequence
          while itemset_index < len(proj_sequence):   # Iterate through itemsets in sequence
            cur_itemset = proj_sequence[itemset_index]    # Current itemset in sequence
            
            if self.has_item(cur_itemset, '_'):    # Add the items following '_' to template 1 list if it's a {_, x} match
              template_1_items += cur_itemset[1:]
            # Itemset doesn't contain '_', check for other matches
            else:
              if self.contains_in_order(cur_itemset, prefix_last_itemset):    # Check if current itemset contains last itemset of prefix i.e {30, x} match
                template_1_items += cur_itemset[cur_itemset.index(prefix_last_itemset[-1])+1:]    # Add the items following prefix's last itemset's last item from the current itemset 
              template_2_items += cur_itemset  # Add all the items in current itemset to template 2 list i.e {30}{x} match
              
            itemset_index += 1
          
          # Add only the unique elements from both lists of current sequence to the main lists as each element is considered only once for a sequence
          all_template_1_items += list(set(template_1_items))   
          all_template_2_items += list(set(template_2_items))
      
        if self.bDEBUG:print "Template 1 items:", all_template_1_items
        if self.bDEBUG: print "Template 2 items:", all_template_2_items
        
        # Compute the total occurences of each element for each template i.e number of sequences it satisfied for a template match
#         print Counter(item for item in all_template_1_items)
        dict_template_1 = dict(Counter(item for item in all_template_1_items))
        dict_template_2 = dict(Counter(item for item in all_template_2_items))
        
        if self.bDEBUG:print "Template 1 item support:", dict_template_1
        if self.bDEBUG:print "Template 2 item support:", dict_template_2 
        
        freq_sequential_patterns = []  # Initialize empty list to contain obtained sequential patterns
        
        # For both the template matches, generate freuqent sequential patterns i.e patterns having support count >= MIS count
        for item, sup_count in dict_template_1.iteritems():
          if sup_count >= mis_count:
            # Add the item to the last itemset of prefix for obtaining the pattern
            freq_sequential_patterns.append((prefix[:-1] + [prefix[-1] + [item]], sup_count))    # Add the pattern with its support count to frequent patterns list
        
        for item, sup_count in dict_template_2.iteritems():
          if sup_count >= mis_count:
            # Append the item contained in a new itemset to the prefix
            freq_sequential_patterns.append((prefix + [[item]], sup_count))    # Add the pattern with its support count to frequent patterns list
        
    #    print "Sequential Patterns Before SDC:", [pattern for pattern, sup_count in freq_sequential_patterns]  
        freq_sequential_patterns = [(pattern, sup_count) for pattern, sup_count in freq_sequential_patterns if self.is_sequence_sdc_satisfied(list(set(itertools.chain(*pattern))))]
    #    print "Sequential Patterns After SDC:", [pattern for pattern, sup_count in freq_sequential_patterns]  
            
        for (seq_pattern, sup_count) in freq_sequential_patterns:   # Iterate through patterns obtained
          if self.has_item(seq_pattern, base_item):    # Add the pattern to the output list if it contains base item
            output_patterns.append((seq_pattern, sup_count))    
          self.prefix_span(output_patterns,seq_pattern, item_sequences, base_item, mis_count)  # Call prefix_span recursively with the pattern as prefix
          
        
    def compute_projected_database(self,prefix, item_sequences, base_item, mis_count):
      projected_db = []
      
      # Populate the projected database with projected sequences
      for sequence in item_sequences:
        cur_pr_itemset = 0
        cur_sq_itemset = 0
        
        while cur_pr_itemset < len(prefix) and cur_sq_itemset < len(sequence):
          if self.contains_in_order(sequence[cur_sq_itemset], prefix[cur_pr_itemset]):    # Sequence itemset contains the prefix itemset, move to next prefix itemset
            cur_pr_itemset += 1
            if cur_pr_itemset == len(prefix): break   # All prefix itemsets are present in the sequence
          
          cur_sq_itemset += 1   # Move to next sequence itemset
        if cur_pr_itemset == len(prefix):   # Prefix was present in the current sequence
          projected_sequence = self.project_sequence(prefix[-1][-1], sequence[cur_sq_itemset:])  # Get the projected sequence
          
          if projected_sequence:    # Non-empty sequence, add to projected database
            projected_db.append(projected_sequence)
      
      if self.bDEBUG:print "DB1:",projected_db
      if self.bDEBUG:print "\n".join([str(sequence) for sequence in projected_db])
      
      # Check if any frequent items are left
      validation_db = self.remove_empty_elements([[[item for item in itemset if not item == '_'] for itemset in sequence] for sequence in projected_db])
      
      if validation_db:   # Non-empty database
    #    projected_db = sdc_filter(projected_db)  # Remove sequences that do not satisfy SDC
        return self.remove_empty_elements(projected_db)    # Remove empty elements and return the projected database
      
      else:   # Empty database
        return validation_db
        
        
    def project_sequence(self,prefix_last_item, suffix):
      suffix_first_itemset = suffix[0]
      if prefix_last_item == suffix_first_itemset[-1]:    # Template 2 projection, return suffix - current_itemset 
        return suffix[1:]
      else:   # Template 1 projection, remove items from first itemset of suffix that are before the index of prefix's last item and put '_' as first element
        suffix[0] = ['_'] + suffix_first_itemset[suffix_first_itemset.index(prefix_last_item)+1:]
        return suffix
    
    
    def support_count(self,sequences, req_item):
      flattened_sequences = [ list(set(itertools.chain(*sequence))) for sequence in sequences ]
      support_counts = dict(Counter(item for flattened_sequence in flattened_sequences for item in flattened_sequence))
      
      return support_counts.get(req_item)
    
      
    def contains(self,big, small):
      return len(set(big).intersection(set(small))) == len(small)
    
    
    def contains_in_order(self,sq_itemset, pr_itemset):
      if(self.contains(sq_itemset, pr_itemset)):
        cur_pr_item = 0
        cur_sq_item = 0
        
        while cur_pr_item < len(pr_itemset) and cur_sq_item < len(sq_itemset):
          if pr_itemset[cur_pr_item] == sq_itemset[cur_sq_item]:
            cur_pr_item += 1
            if cur_pr_item == len(pr_itemset): return True
          
          cur_sq_item += 1
       
      return False 
            
    
    def has_item(self,source_list, item):
      if source_list:
        while isinstance(source_list[0], list):
          source_list = list(itertools.chain(*source_list))
        
        return item in source_list  
      return False
    
    
    
    def is_sequence_sdc_satisfied(self,sequence):
      if not sequence:
        return False
      
      if len(sequence) > 1:
        for item1 in sequence:
          sup_item1 = self.actual_supports.get(item1)
          for item2 in sequence:
            if not item1 == '_' and not item2 == '_' and not item1 == item2:
              if abs(sup_item1 - self.actual_supports.get(item2)) > self.sdc:
                return False 
      
      return True  
    
    
    def remove_infrequent_items(self,item_sequences, min_support_count):
      # Get the support count for each item in supplied sequence database
      flattened_sequences = [ list(set(itertools.chain(*sequence))) for sequence in item_sequences ]
      support_counts = dict(Counter(item for flattened_sequence in flattened_sequences for item in flattened_sequence))
      
      # Remove the infrequent items from the sequence database  
      filtered_item_sequences = [[[item for item in itemset if support_counts.get(item) >= min_support_count or item == '_']for itemset in sequence] for sequence in item_sequences]
      
      return self.remove_empty_elements(filtered_item_sequences)    # Return the new sequence database
        
    
    def remove_empty_elements(self,source_list):
      filtered_list = []    # Declare list to contain filter results
      
      # Check to see if list has sub lists as items   
      if source_list and isinstance(source_list[0], list):
            
        for child_list in source_list:
          filtered_child_list = self.remove_empty_elements(child_list)    # Recurse for filtering each child_list
          
          if filtered_child_list:   # Append only non-empty lists
            filtered_list.append(filtered_child_list)
      
      else:
        filtered_list = source_list 
       
      return filtered_list    # Return the filtered list from current recursion

  
if __name__ == '__main__':
    main()
     