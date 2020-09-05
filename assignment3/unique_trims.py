import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: dna_trimmed
    # value: 1
    key = record[1]
    key = key[:-10]
    mr.emit_intermediate(key, 1)

def reducer(key, list_of_friends):
    # key: dna_trimmed
    mr.emit(key)
    

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
