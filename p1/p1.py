import sys 
import math
import string
import re 
from collections import defaultdict

# EXTERNAL MODULE START 
###############################################################
# This code is taken from the munkres module by brian clapper
# the copyright and the licence agreeement follows
"""
Copyright and License
=====================

This software is released under a BSD license, adapted from
<http://opensource.org/licenses/bsd-license.php>

Copyright (c) 2008 Brian M. Clapper
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name "clapper.org" nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


class Munkres:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.marked = None
        self.path = None

    def make_cost_matrix(profit_matrix, inversion_function):
        """
        **DEPRECATED**

        Please use the module function ``make_cost_matrix()``.
        """
        import munkres
        return munkres.make_cost_matrix(profit_matrix, inversion_function)

    make_cost_matrix = staticmethod(make_cost_matrix)

    def pad_matrix(self, matrix, pad_value=0):
        """
        Pad a possibly non-square matrix to make it square.

        :Parameters:
            matrix : list of lists
                matrix to pad

            pad_value : int
                value to use to pad the matrix

        :rtype: list of lists
        :return: a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [0] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[0] * total_rows]

        return new_matrix

    def compute(self, cost_matrix):
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of (row, column) tuples
        that can be used to traverse the matrix.

        :Parameters:
            cost_matrix : list of lists
                The cost matrix. If this cost matrix is not square, it
                will be padded with zeros, via a call to ``pad_matrix()``.
                (This method does *not* modify the caller's matrix. It
                operates on a copy of the matrix.)

                **WARNING**: This code handles square and rectangular
                matrices. It does *not* handle irregular matrices.

        :rtype: list
        :return: A list of ``(row, column)`` tuples that describe the lowest
                 cost path through the matrix

        """
        self.C = self.pad_matrix(cost_matrix)
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)
        self.marked = self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.marked[i][j] == 1:
                    results += [(i, j)]

        return results

    def __copy_matrix(self, matrix):
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n, val):
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self):
        """
        For each row of the matrix, find the smallest element and
        subtract it from every element in its row. Go to Step 2.
        """
        C = self.C
        n = self.n
        for i in range(n):
            minval = min(self.C[i])
            # Find the minimum value for this row and subtract that minimum
            # from every element in the row.
            for j in range(n):
                self.C[i][j] -= minval

        return 2

    def __step2(self):
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.C[i][j] == 0) and \
                   (not self.col_covered[j]) and \
                   (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True

        self.__clear_covers()
        return 3

    def __step3(self):
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to DONE, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1:
                    self.col_covered[j] = True
                    count += 1

        if count >= n:
            step = 7 # done
        else:
            step = 4

        return step

    def __step4(self):
        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. Save the smallest uncovered value and Go to Step 6.
        """
        step = 0
        done = False
        row = -1
        col = -1
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero()
            if row < 0:
                done = True
                step = 6
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self):
        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self):
        """
        Add the value found in Step 4 to every element of each covered
        row, and subtract it from every element of each uncovered column.
        Return to Step 4 without altering any stars, primes, or covered
        lines.
        """
        minval = self.__find_smallest()
        for i in range(self.n):
            for j in range(self.n):
                if self.row_covered[i]:
                    self.C[i][j] += minval
                if not self.col_covered[j]:
                    self.C[i][j] -= minval
        return 4

    def __find_smallest(self):
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxint
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval

    def __find_a_zero(self):
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = 0
        n = self.n
        done = False

        while not done:
            j = 0
            while True:
                if (self.C[i][j] == 0) and \
                   (not self.row_covered[i]) and \
                   (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j += 1
                if j >= n:
                    break
            i += 1
            if i >= n:
                done = True

        return (row, col)

    def __find_star_in_row(self, row):
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col):
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self, path, count):
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1

    def __clear_covers(self):
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self):
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def make_cost_matrix(profit_matrix, inversion_function):
    """
    Create a cost matrix from a profit matrix by calling
    'inversion_function' to invert each value. The inversion
    function must take one numeric argument (of any type) and return
    another numeric argument which is presumed to be the cost inverse
    of the original profit.

    This is a static method. Call it like this:

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    For example:

    .. python::

        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxint - x)

    :Parameters:
        profit_matrix : list of lists
            The matrix to convert from a profit to a cost matrix

        inversion_function : function
            The function to use to invert each entry in the profit matrix

    :rtype: list of lists
    :return: The converted matrix
    """
    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix, msg=None):
    """
    Convenience function: Displays the contents of a matrix of integers.

    :Parameters:
        matrix : list of lists
            Matrix to print

        msg : str
            Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print msg

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            width = max(width, int(math.log10(val)) + 1)

    # Make the format string
    format = '%%%dd' % width

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            sys.stdout.write(sep + format % val)
            sep = ', '
        sys.stdout.write(']\n')


##################################################################
# EXTERNAL MODULE END 

# DEBUG 
#DEBUG = True
DEBUG = False

# GLOBAL VARS 
prime_list = [] 
vowels = 'aeiouy'
consonants = 'bcdfghjklmnpqrstvwxz'
all_lttrs = string.letters 
# check for prime number
# only works for prime numbers >=5 
def is_prime(num):
	if (num % 2 == 0) or (num % 3 == 0) :
		return False 
	# basic algo says check from that number to square root of that num
	# but we will follow the 6n+1/-1  rule 
	# the rule says that all primes numbers (except 2,3) can be rep as 6n+/-1
	n = 1
	# this condition will result in one extra check for e.g for 35 we will check for 5 < (sqrt(35)  and 7 > (sqrt(35)) 
	# special case 25 (5 square),11 square = 121 
	# make sure u check for div by 5 		
	# this does not check for divisors 2,3 
	while( 6*n <= (math.sqrt(num)+1)):
		# test if 6n +1 or 6n-1 are factors of num  
		if ((num % (6*n+1)) == 0 ):
				return False
		elif ((num % (6*n-1))== 0 ):
				return False 
		# update n for next iteration 	
		n = n + 1
	return True 		
	
# generate all prime numbers between 2 and 500 store it in a list 
def get_primes():
	# add 2,3 to list 
	prime_list.append(2)
	prime_list.append(3)
	for num in xrange(4,500):
		if(is_prime(num)):
			prime_list.append(num)
		#print prime_list

# get the vowel score 
def get_v_count(cust):

	cust_lower = cust.lower()
	v_count = 0 
	for vowel in vowels:
		l = re.findall(vowel,cust_lower)
		v_count += len(l)
	return v_count

def get_c_count(cust):

	cust_lower = cust.lower()
	c_count = 0 
	for cons in consonants:
		l = re.findall(cons,cust_lower)
		c_count += len(l)
	return c_count

# read all customers and store all their info in a dict
def get_custs(custs):
	# init a dict of list
	cust_dict = defaultdict(list)
	for cust in custs:
		cust_len = get_num_lttrs(cust)
		v_score = get_v_count(cust)*1.5
		c_score = get_c_count(cust)
		cust_dict[cust] = [v_score,c_score,cust_len]
	return cust_dict

# get num letters 
def get_num_lttrs(prod):
	lttr_count = 0
	prod = prod.lower()
	# algo for counting letters = search for all lttrs using regex  
	for lttr in all_lttrs:
		l = re.findall(lttr,prod)
		lttr_count += len(l)
	return lttr_count 	

# returns true if the given input has common factors 
def has_common_factors(num_k,num_prod):
	if DEBUG:
 		print "looking for common factors for ",num_k,num_prod
	# using Eucledians algorithm
	if(num_prod > num_k):
		dividend = num_prod
		divisor = num_k 
	elif(num_k > num_prod):	
		dividend = num_k
		divisor = num_prod 
	elif(num_k == num_prod):
		# if they are equal then they have a common factor 
		return True
	# the algo 
	condition = True
	while(condition):
		# divide bigger num by smaller 
		rem = dividend % divisor 
		if(rem == 1):
			# no common factor found 
			condition = False 
			ret_val = False 
		elif(rem == 0):
			# common factor found 
			condition = False
			ret_val = True
		else:
			# reset dividend and divisor 
			dividend = divisor 
			divisor = rem 
			

	return ret_val 

# calculate the matrix 
def get_matrix(cust_dict,prod_l):
	main_matrix = [] 
	# cust_remaining list 
	#cust_l = cust_dict.keys()
	# get a matrix prod * customers = rectangular or square
	for prod in prod_l :
		if DEBUG:
			print "working with prod",prod
		# for this prod find all the customer values
		# first find if the number of letters in the prod 
		num_lp = get_num_lttrs(prod)
		if(num_lp % 2 == 0):
			# even 
			# cust dict contains even score at 0th index
			if DEBUG:
				print "this prod has even letters= ",num_lp
			score_index = 0 
		else :
			# odd 
			# cust dict contains even score at 1th index
			if DEBUG:
				print "this prod has odd letters= ",num_lp
			score_index = 1 
		# iterate all customers 
		temp_list = []
		for k,v in cust_dict.iteritems():

			
			if DEBUG:
				print "workind with cust",k,"and val_list",(v,)

			init_score =  v[score_index]
			# does this have any common factors 
			# get num of letters in customer 
			num_lk = v[2]
			var_bool = has_common_factors(num_lk,num_lp)
			if(var_bool):
				init_score = init_score * 1.5
				if DEBUG:
					print "this pair has a common factor score =",init_score
			else:	
				if DEBUG:
					print "no common factor score =",init_score
			temp_list.append(init_score)	
		# add this temp list to main_matrix 
		main_matrix.append(temp_list)
	if DEBUG:
		print main_matrix 	
	return main_matrix 

#this is the main method 
def process_input(txt):
	ans_lst = []
	lines = txt.split('\n')
	for line in lines:
	
		if DEBUG:
			print line 
		line = line.strip()
		line = line.strip('\n')
		if line:
			custs,prods = line.split(';')
			# make a cust list 
			custs = custs.split(',')
			# precompute custs list 
			cust_dict = get_custs(custs)
			# make a prod list 
			prod_l = prods.split(',') 
			# get suitability score 
			line_mat = get_matrix(cust_dict,prod_l)
			# Calculate the total score 
			cost_matrix = make_cost_matrix(line_mat,lambda cost: sys.maxint-cost)
			hungarian_m = Munkres()
			indexes = hungarian_m.compute(cost_matrix)
			line_score = 0
			for row,column in indexes:
				value = line_mat[row][column]
				line_score += value
			# TODO TWO DECIMAL PLACES

			if DEBUG:
				print "final value",line_score	
			ans_lst.append(line_score)

	return ans_lst

def read_file(filename):
	fin = open(filename)
	txt = fin.read()
	fin.close()
	return txt 

if __name__ == "__main__":
	get_primes()
	if(len(sys.argv) == 2):
		txt = read_file(sys.argv[1])
	else:
		print "invalid cmd line arg, please enter one file as ip "
		sys.exit() 
	ans_list = process_input(txt)
	for ans in ans_list:
		print "%.2f"%(ans)
	sys.exit(0)	

