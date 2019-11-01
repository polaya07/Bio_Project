import numpy as np
import pandas as pd
import glob
from Bio import SeqIO
import sys

seqs = []
genes = []
ids = []

#iterate through fna files
for i,file in enumerate(list(glob.glob('../data_genes/*.fna'))):                                            
    for j,seq_record in enumerate(SeqIO.parse(file, "fasta")):

        #extract entries with valid labels
        desc = seq_record.description
        if '[pseudo=true]' in desc:
            label = 0
        elif '[protein=hypothetical protein]' in desc:
            continue
        else:
            label = 1
        
        ids.append(seq_record.id)
        seqs.append(str(seq_record.seq))
        genes.append(label)
        
        sys.stdout.write("processing file %i, entry %i     \r" % (i,j))
        sys.stdout.flush()
        
print()

#create dataframe
df = pd.DataFrame({'id':ids,'seq':seqs,'gene':genes})
df.to_csv('../data_genes/gene_dataset.csv')

#data statistics
print(df)
print(df.groupby(['gene']).agg(['count']))