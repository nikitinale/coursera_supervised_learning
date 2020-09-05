import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: two person
    # value: friend
    impo = sorted(record)
    key = impo[0]+impo[1]
    friend = [record[1], record[0]]
    mr.emit_intermediate(key, friend)

def reducer(key, list_of_friends):
    # key: person
    # value: friend
    if len(list_of_friends) == 1 :
        mr.emit((list_of_friends[0][0], list_of_friends[0][1]))
        mr.emit((list_of_friends[0][1], list_of_friends[0][0]))
    
# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
