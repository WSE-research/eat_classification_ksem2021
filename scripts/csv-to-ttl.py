import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    filename = args.filename
    
    dataset = pd.read_csv(filename, sep=';')

    for i, row in dataset.iterrows():
        triple_1 = "<{id_}> rdf:type <urn:question> .\n".format(id_=row.question)
        triple_2 = "<{id_}> rdfs:label \"{text}\" .\n".format(id_=row.question, text=row.questionText.replace('"','\\"').replace("'",'\\"'))
        triple_3 = "<{id_}> <urn:class> <{class_}> .\n".format(id_=row.question, class_=row.type)
    
        with open("{0}-csv-to-ttl.ttl".format(filename), "a") as myfile:
            myfile.write(triple_1)
            myfile.write(triple_2)
            myfile.write(triple_3)

    print("Finished Dataset Conversion")
