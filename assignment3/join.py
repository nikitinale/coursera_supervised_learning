import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: id
    # value: content
    key = record[1]
    content = record[:]
    mr.emit_intermediate(key, content)

def reducer(key, list_of_records):
    # key: id
    # value: id_document
    recs = []
    list2 = list_of_records[:]
    for r1 in list_of_records :
        if r1[0] ==  "order" :
            for r2 in list2 :
                if r2[0] == "line_item" :
                    line = r1+r2
                    mr.emit((line))

    #mr.emit((line))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)
