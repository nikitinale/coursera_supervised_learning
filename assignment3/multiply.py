import MapReduce
import sys

#mr_pred = MapReduce.MapReduce()
mr = MapReduce.MapReduce()
"""
Word Count Example in the Simple Python MapReduce Framework
"""
D = {"a":5, "b":5}

def p_mapper(record) :
    key = record[0]
    if key == 'b' :
        value = record[2] # rows in b
    elif key == 'a' :
        value = record[1] # columns in a
    mr_pred.emit_intermediate(key, value)

def p_reducer(key, values):
    D[key] = max(values)+1

# =============================
# Do not modify above this line

def mapper(record):
    # key: document identifier
    # value: document contents
    if record[0] == 'a' : 
        for i in range(D["b"]) :
            key = str(record[1])+'-'+str(i)
            value = [record[0], record[2], record[3]]
            mr.emit_intermediate(key, value)
    if record[0] == 'b' : 
        for i in range(D["a"]) :
            key = str(i)+'-'+str(record[2])
            value = [record[0], record[1], record[3]]
            mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: word
    # value: list of occurrence counts
    total = 0
    i, j = key.split("-")
    second_list = list_of_values[:]
    for v in list_of_values:
        if v[0] == 'a' :
            for w in second_list :
                if w[0] == 'b' and w[1] == v[1] :
                    total += v[2]*w[2]
    mr.emit((int(i), int(j), total))

# Do not modify below this line
# =============================
if __name__ == '__main__':
    inputdata = open(sys.argv[1])
    #mr_pred.execute(inputdata, p_mapper, p_reducer)
    #inputdata.close()
    #inputdata = open(sys.argv[1])
    mr.execute(inputdata, mapper, reducer)
