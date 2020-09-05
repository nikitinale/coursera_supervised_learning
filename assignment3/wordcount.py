import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    key = (record[1], record[2])
    value = {(record[0], record[2])
    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    total = 0
    for v in list_of_values:
      total += 1
      mr.emit((key, v))

def main() :
    mr = MapReduce.MapReduce()
    inputdata = open(sys.argv[1])
    mr.execute(inputdata, mapper, reducer)
    

# Do not modify below this line
# =============================
if __name__ == '__main__':
    main()
