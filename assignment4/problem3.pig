register s3n://uw-cse-344-oregon.aws.amazon.com/myudfs.jar

-- load the test file into Pig
raw = LOAD 's3n://uw-cse-344-oregon.aws.amazon.com/cse344-test-file' USING TextLoader as (line:chararray);
-- later you will load to other files, example:
--raw = LOAD 's3n://uw-cse-344-oregon.aws.amazon.com/btc-2010-chunk-000' USING TextLoader as (line:chararray); 

-- parse each line into ntriples
ntriples = foreach raw generate FLATTEN(myudfs.RDFSplit3(line)) as (subject:chararray,predicate:chararray,object:chararray);

--!filter subject matches '.*rdfabout\.com.*' | subject matches '.*business.*'
filtered = FILTER ntriples BY (subject matches '.*rdfabout\.com.*');

-- copy of DB
second = foreach filtered generate $0 as subject2, $1 as predicate2, $2 as object2;

--!JOIN two copies: object=subject2 | subject=subject2
joined = JOIN filtered BY subject, second BY subject2 PARALLEL 50;

--remove dublicates
final = DISTINCT joined PARALLEL 50;

-- store the results in the folder /user/hadoop/problem-3
store final into '/user/hadoop/problem-3A' using PigStorage();
