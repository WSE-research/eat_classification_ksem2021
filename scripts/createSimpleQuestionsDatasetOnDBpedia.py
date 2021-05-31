# -*- coding: utf-8 -*-

# run the following in your shell before executing the Python command
# export PYTHONIOENCODING=utf8 

"""
    not working for question: 75906 -> "gender male" rdf:type dbo:Book
                    question: 18 -> "USA" rdf:type dbo:Person
    observation: primary types are represented including their supertypes, secondary (false?) types are represented without supertypes
    
    Create a statistics via SPARQL 
        SELECT ?number (COUNT(?question) AS ?count) {
            ?question <urn:SimpleQuestions:NumberOfDBpediaTopLevelTypeFound> ?number
        }
        GROUP BY ?number
        ORDER BY ?number
"""

import sys
import GstoreConnector
import json
from pprint import pprint
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

minnumber = 10001 # from ID (including)
maxnumber = 11000 # to ID (including)
outfilename = "simplequestionsOnDBpedia/output"+str(minnumber)+".ttl"
dbpedia = SPARQLWrapper("http://dbpedia.org/sparql")
# infile = "../data/SimpleQuestions/original/annotated_fb_data_train.txt"
# infile = "../data/SimpleQuestions/original/annotated_fb_data_test.txt"
infile = "../data/SimpleQuestions/original/annotated_fb_data_valid.txt"



def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return list(dict.fromkeys(lst3)) 


def readAcceptedDBpediaTypes():
    acceptedTypes = []
    with open("topLevelDBpediaTypes.txt", 'r', errors='replace') as f:
        for line in f.readlines():
            acceptedTypes.append(line.strip())
    return acceptedTypes


def replaceFreebaseUrlPrefixByGdataUrlPrefix(url):
    return url.replace("www.freebase.com/m/","http://rdf.freebase.com/ns/m.")


def readSimpleQuestionInputFile(filename):
    input = []
    with open(filename, 'r', errors='replace') as f:
        for line in f.readlines():
            fragments = line.split("\t")
            input.append({
                    "s":fragments[0], 
                    "p":fragments[1], 
                    "o":fragments[2], 
                    "question":fragments[3]
                })
            #pprint(input)
    return input 


def getTypesOfResource(dbpedia, resource):
    types = []
    # see comments at the beginning of the file w.r.t. this query
    sparql = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?s ?type WHERE {
  ?s owl:sameAs <%s> .
  
  {
    ?s rdf:type ?type .
  }
  UNION
  {
    ?s rdf:type ?intermediatype .
    FILTER( CONTAINS(str(?intermediatype),"http://dbpedia.org/ontology"))
    ?intermediatype rdfs:subClassOf ?type .
  }
  UNION
  {
    ?s rdf:type ?intermediatype .
    FILTER( CONTAINS(str(?intermediatype),"http://dbpedia.org/ontology"))
    ?intermediatype rdfs:subClassOf ?intermediatype2 .
    ?intermediatype2 rdfs:subClassOf ?type .
  }
  UNION
  {
    ?s rdf:type ?intermediatype .
    FILTER( CONTAINS(str(?intermediatype),"http://dbpedia.org/ontology"))
    ?intermediatype rdfs:subClassOf ?intermediatype2 .
    ?intermediatype2 rdfs:subClassOf ?intermediatype3 .
    ?intermediatype3 rdfs:subClassOf ?type .
  }
  
  FILTER( CONTAINS(str(?type),"http://dbpedia.org/ontology"))
}     
    """ % (replaceFreebaseUrlPrefixByGdataUrlPrefix(resource))
    
    # we use this query
    sparql = """
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?s ?type WHERE {
  ?s owl:sameAs <%s> .
  ?s rdf:type ?type .
  FILTER( CONTAINS(str(?type),"http://dbpedia.org/ontology"))
}     
    """ % (replaceFreebaseUrlPrefixByGdataUrlPrefix(resource))

    #print(sparql)
    
    dbpedia.setQuery(sparql)
    dbpedia.setReturnFormat(JSON)
    result = dbpedia.query().convert()
    #result = {}
    
    #pprint(result)
    dbpediaResource = None
    for item in result.get("results").get("bindings"):
        dbpediaResource = item.get("s").get("value")
        types.append(item.get("type").get("value"))
    return dbpediaResource,types


input = readSimpleQuestionInputFile(infile)
outfile = open(outfilename, 'w', errors="surrogateescape", encoding="utf-8")
answerrelation = "urn:SimpleQuestions:hasAnswerResource"
acceptedTypes = readAcceptedDBpediaTypes()

counter = 0
countEmptyTypes = 0
countProcessed = 0
countNoDBpediaTypeFound = 0
countNoToplevelDBpediaTypeFound = 0
countMultipleToplevelDBpediaTypeFound = 0
topLevelDBpediaTypeRelation = "urn:SimpleQuestions:NumberOfDBpediaTopLevelTypeFound"

for item in input:
    counter+=1
    if counter >= minnumber and counter <= maxnumber:
        countProcessed += 1
        question = item.get("question")
        resource = item.get("o")
        print("\n%5d. %s (%s)" % (counter, question.strip(), resource))
        dbpediaResource, types = getTypesOfResource(dbpedia, resource)
        questionid = "urn:SimpleQuestions:number:%d" % (counter)
        
        print(dbpediaResource)
        pprint(types)
        
        foundToplevelDBpediaTypes = intersection(types, acceptedTypes)
        
        output = """
# %d: %s 
<%s> <%s> <http://%s> .
<%s> <http://www.w3.org/2000/01/rdf-schema#comment> "%s"^^xsd:string .
<http://%s> <http://www.w3.org/2002/07/owl#sameAs> <%s> .
        """ % (
                counter,
                question,
                questionid,
                answerrelation,
                resource,
                questionid,
                question.strip().replace('"','\\"'),
                resource,
                replaceFreebaseUrlPrefixByGdataUrlPrefix(resource)
        )
        
        if len(types) > 0:
            output += """
<http://%s> <http://www.w3.org/2002/07/owl#sameAs> <%s>.
<%s> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <%s> . 
           """ % (
                resource,
                dbpediaResource,
                dbpediaResource,
                ">,\n\t\t\t<".join(types)
            )

        if len(foundToplevelDBpediaTypes) == 0:
            if len(types) == 0:
                countNoDBpediaTypeFound += 1
                print("NO DBPEDIA TYPES FOUND.")
            else:
                countNoToplevelDBpediaTypeFound += 1
                print("NO TOPLEVEL DBPEDIA TYPES FOUND:")
                print(types)
        elif len(foundToplevelDBpediaTypes) == 1:
            pass # expected default
        else:
            countMultipleToplevelDBpediaTypeFound += 1
            print("MULTIPLE TOPLEVEL DBPEDIA TYPES FOUND: ")
            pprint(foundToplevelDBpediaTypes)
        output += """
<%s> <%s> "%d"^^xsd:int .
        """ % (
            questionid,
            topLevelDBpediaTypeRelation,
            len(foundToplevelDBpediaTypes)
        )


        outfile.write(output)


statistics = "%d question, %d NO topDBpediaType found (%2.0d %%), %d NO topLevelDBpediaType found (%2.0d %%), %d MULTIPLE topLevelDBpediaType found (%2.0d %%)" % (
        countProcessed, 
        countNoDBpediaTypeFound, (countNoDBpediaTypeFound/countProcessed*100),
        countNoToplevelDBpediaTypeFound, (countNoToplevelDBpediaTypeFound/countProcessed*100),
        countMultipleToplevelDBpediaTypeFound, (countMultipleToplevelDBpediaTypeFound/countProcessed*100)
)
print(statistics)
outfile.write("# %s" % (statistics))
outfile.close()
