register s3n://uw-cse-344-oregon.aws.amazon.com/myudfs.jar

-- load the test file into Pig
--raw = LOAD 's3n://uw-cse-344-oregon.aws.amazon.com/cse344-test-file' USING TextLoader as (line:chararray);
-- later you will load to other files, example:
--raw = LOAD 's3n://uw-cse-344-oregon.aws.amazon.com/btc-2010-chunk-000' USING TextLoader as (line:chararray); 
raw = LOAD 's3n://uw-cse-344-oregon.aws.amazon.com/btc-2010-chunk-*' USING TextLoader as (line:chararray); 

-- parse each line into ntriples
ntriples = foreach raw generate FLATTEN(myudfs.RDFSplit3(line)) as (subject:chararray,predicate:chararray,object:chararray);

--group the n-triples by object column
subjects = group ntriples by (subject) PARALLEL 100;

-- flatten the subjects out (because group by produces a tuple of each object
-- in the first column, and we want each object ot be a string, not a tuple),
-- and count the number of tuples associated with each object
count_by_subjects = foreach subjects generate flatten($0), COUNT($1) as count PARALLEL 100;

--group count_objects by number of counts
counts = group count_by_subjects by (count) PARALLEL 100;

--calculate subject X n_counts hist
hist_counts = foreach counts generate flatten($0), COUNT($1) as count PARALLEL 100;

--order the resulting tuples by their count in descending order
ordered_hist = order hist_counts by (count) PARALLEL 100;

-- store the results in the folder /user/hadoop/problem-4
store ordered_hist into '/user/hadoop/problem-4' using PigStorage();
