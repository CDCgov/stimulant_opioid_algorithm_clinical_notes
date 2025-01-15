# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:13:52 2024
Code to detect non-therapeutic stimulant and opioid use in clinical notes
of either free text or xml-based FHIR format.
@author: Nikki Adams
"""

import configparser
from pathlib import Path
import re


import numpy as np
#pandas needs 2.0 or higher
import pandas as pd
import pyodbc
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from build_queries import Query

import warnings
warnings.simplefilter('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', message = 'This pattern is interpreted as a regular expression,')
warnings.filterwarnings('ignore', message = 'pandas only supports SQLAlchemy connectable')

configfile = "config.txt"
chunksize = 1000

config = configparser.ConfigParser()
config.read(configfile)
required_headers = ["INPUT_SETTINGS", "SEARCH_SETTINGS", "OUTPUT_SETTINGS"]
for req in required_headers:
    if req not in config:
        raise KeyError(f"Required configuration option {req} not in config file. See ReadMe for guidance")

input_type = config['INPUT_SETTINGS']['input_type']
text_format = config['INPUT_SETTINGS']['text_format']
model_indir = config['INPUT_SETTINGS']['model_indir']
if model_indir == '' and text_format == 'free text':
    raise ValueError("Free text is specified but a model path or location is not specified. Free text analysis requires model. See ReadMe for guidance")
cnxn_string = config['INPUT_SETTINGS']['cnxn_string']
sql_query = config['INPUT_SETTINGS']['sql_query']
infile_path = config['INPUT_SETTINGS']['infile_path']
search_terms_path = config['INPUT_SETTINGS']['search_terms_path']

col_to_search = config['SEARCH_SETTINGS']['col_to_search']
note_type_col = config['SEARCH_SETTINGS']['note_type_col']
drugscreen_note_type = config['SEARCH_SETTINGS']['drugscreen_note_type']
meds_note_type = config['SEARCH_SETTINGS']['meds_note_type']
ehr_diag_titles = config['SEARCH_SETTINGS']['ehr_diag_titles']

results_path = config['OUTPUT_SETTINGS']['results_path']
cols_to_keep = config['OUTPUT_SETTINGS']['cols_to_keep']
group_cols = config['OUTPUT_SETTINGS']['group_cols']


if infile_path == '' and input_type.upper() in ('CSV', 'SAS'):
    raise ValueError("Input type has been specified as CSV or SAS but no input file path has been specified. See ReadMe")
infile_path = Path(infile_path)
if search_terms_path == '':
    raise ValueError("Search term file must be specified. See ReadMe")
search_terms_path = Path(search_terms_path)

#Search Spec
if col_to_search == '':
    raise ValueError("Column to search must be specified")
col_to_search = col_to_search.upper()
if text_format == 'FHIR' and (note_type_col == '' or drugscreen_note_type == '' or meds_note_type == '' or ehr_diag_titles == ''):
    raise ValueError("Text format has been specified as FHIR. Note type column, as well as values in that column indication medications, lab results, \
    and diagnosis-likely values must be specified")
note_type_col = note_type_col.upper()
#presumption is that there is one note type each for medications and labs (string), but can be more than one for diagnoses type (list)
ehr_diag_titles = [x.strip() for x in ehr_diag_titles.split(',')]

#Output Spec
if results_path == '':
    raise ValueError("An output path for results must be specified. See ReadMe")
results_path = Path(results_path)
if cols_to_keep == '':
    raise ValueError("Please specify at least one column to keep in the output, such as a linkage key / unique identifier")
cols_to_keep = [x.strip() for x in cols_to_keep.split(",")]

if group_cols == '':
    group_cols = None
else:
    group_cols = [x.strip() for x in group_cols.split(",")]
    
#Read in search terms

search_terms = pd.read_excel(Path(search_terms_path), sheet_name = None)
stim_term_df = search_terms["STIMULANTS"]     
opioid_term_df = search_terms['OPIOIDS']     

all_terms_df = pd.concat([stim_term_df, opioid_term_df], sort = False)

all_terms_df['Term'] = all_terms_df['Term'].str.strip().str.lower()
all_terms_df['Category'] = all_terms_df['Category'].str.strip().str.upper()
all_terms_df = all_terms_df[['Term', 'Category']].copy()
if all_terms_df['Term'].nunique() != all_terms_df.drop_duplicates().shape[0]:
    print('Warning: One or more of the same term appears with more than one category. Only one category will be mapped for each term')
    
#some combinations are so indicative of non-therapeutic use, 
#you're better off assuming when you see them that that is what they flag

overrides_d = {'Term':[], 'Category':[]}

for _, row in all_terms_df.iterrows():
    k = row.Term
    v = row.Category
    if v =='RX_OPIOD':
        new_v = 'OPIOID_MISUSE_OVERRIDE'
    elif v == 'RX_AMPHETAMINE':
        new_v = 'STIMULANT_MISUSE_OVERRIDE'
    elif v == 'MAT':
        new_v = 'OPIOID_MISUSE_OVERRIDE'
    elif v == 'UNSPECIFIED_STIMULANT':
        new_v = 'STIMULANT_NON_TX_UNSP_OVERRIDE'
    elif v == 'UNSPECIFIED_OPIOID':
        new_v = 'OPIOID_NON_TX_UNSP_OVERRIDE'
    else:
        continue
    
    abuse_k = f"{k} abuse"
    dependence_k = f"{k} dependence" #no override for MAT for dependence
    seeking_k1 = f"{k}-seeking"
    seeking_k2 = f"{k} seeking"
    
    overrides_d['Term'].append(abuse_k)
    overrides_d['Category'].append(new_v)
    
    overrides_d['Term'].append(seeking_k1)
    overrides_d['Category'].append(new_v)
    overrides_d['Term'].append(seeking_k2)
    overrides_d['Category'].append(new_v)    
    if v != 'MAT':
        overrides_d['Term'].append(dependence_k)
        overrides_d['Category'].append(new_v)

overrides = pd.DataFrame(overrides_d)
prev_len = len(all_terms_df)
all_terms_df = all_terms_df[~all_terms_df.Term.isin(overrides['Term'].values)].copy()
# override_count = prev_len - len(all_terms_df)
# print(f"{override_count} rows from original dataframe reclassified due to overrides")       
all_terms_df = pd.concat([all_terms_df, overrides])

#In this version of the algorithm, we will not distinguish between different types of prescription stimuulant misuse
#That distinction is collapsed here for simplicity further down in the code
all_terms_df['Category'] = all_terms_df['Category'].str.replace('RX_AMPHETAMINE', 'RX_STIM')
all_terms_df['Category'] = all_terms_df['Category'].str.replace('LISDEXAMFETAMINE', 'RX_STIM') 
all_terms_df['Category'] = all_terms_df['Category'].str.replace('METHYLPHENIDATE', 'RX_STIM')
all_terms_df['Category'] = all_terms_df['Category'].str.replace('RX_COCAINE', 'RX_STIM')


#Build the regular expression that will be used to search text. These are the same terms in the master term list

query = Query(all_terms_df['Term'].values.tolist(), input_type = "list", 
              query_type = "boundary with s")
drug_regex = query.build_re()

#Create a dictionary that maps the search terms to their categories. Regex was built with automatic optional s for plurals
#so add those to the dictionary as well
cat_d = {all_terms_df['Term'].values[i].lower().strip() : all_terms_df['Category'].values[i].upper().strip() for i in range(len(all_terms_df))}
for term, category in sorted(cat_d.items()):
    if not term.endswith('s'):
        cat_d[term+"s"] = category
        
#This function will normalize the text, replacing term for evaluation with DRUGTERM and removing bad characters
#that were also removed during model training
def normalize_w_drugterm(text, searchterm):

    bad_chars = re.compile(r"Â¶|ï¿½|¶|\?â€¢|â€¢|Â¶â€¢|ï¿½|Â¶")
    text = re.sub(bad_chars, " ", text)
    text = re.sub(r" {3,}", "  ", text)
    match = re.search(searchterm, text, flags = re.IGNORECASE)
    try:
        text = re.sub(match.group(), "DRUGTERM", text, count = 1)
    except TypeError:
        raise Exception("Match is: ", match)
    except AttributeError:
        raise Exception(f"Searchterm is: {searchterm} and text is {text}")
    return(text)

#this code is written to tokenize in batches

def tokenize_and_encode(texts):
    max_len = 150
    encoded_dict = tokenizer.batch_encode_plus(
                        texts,                      
                        add_special_tokens = True, 
                        max_length = max_len,    
                        truncation = True, 
                        pad_to_max_length = True,
                        return_attention_mask = True,   #
                        return_tensors = 'pt'    )

    return(encoded_dict)


#evaluate the encoded text (input is dictionary)
def eval_with_label(encoded_dict):
    with torch.no_grad():        
        sample_output= model(encoded_dict['input_ids'].to(device), 
                token_type_ids=None, 
                attention_mask= encoded_dict['attention_mask'].to(device))
        sample_logits = sample_output.logits.detach().cpu().numpy()

        
    return(sample_logits)

#this function extracts a snippet of text around a drug term. Only this shorter snippets is evaluated 
#by the model for free text, rather than the whole text:
def extract_snippets(text, match_object, window):
    snippet_start = max([match_object.start() - window, 0])
    snippet_end = min([match_object.end() + window, len(text)])
    while snippet_start > 0:
        if re.search(r"[^a-z0-9-]", text[snippet_start]) is None:
            snippet_start -= 1
        else:
            break
    while snippet_end < len(text):
        if re.search(r"[^a-z0-9-]", text[snippet_end]) is None:
            snippet_end += 1
        else:
            break
    snippet = text[snippet_start : snippet_end]    
    return((snippet.strip(), match_object.group()))

#this function is for FHIR-standard input. Determinations are made at the row level
#but each row can have multiple flags
def determine_note_flag_FHIR(cats, non_tx):

    flags = set()  
       
    if 'OPIOID_MISUSE_OVERRIDE' in cats:
        flags.add('OPIOID_MISUSE_NLP')
    if 'STIMULANT_MISUSE_OVERRIDE' in cats:
        flags.add('STIM_MISUSE_NLP')
    if 'STIMULANT_NON_TX_UNSP_OVERRIDE' in cats:
        flags.add('STIM_NON_TX_UNSP_NLP')
    if 'OPIOID_NON_TX_UNSP_OVERRIDE' in cats:
        flags.add('OPIOID_NON_TX_UNSP_NLP')
        
        
    if 'COCAINE' in cats and non_tx:
        flags.add('ILLICIT_COCAINE_NLP')

    if 'RX_STIM' in cats: #formerly separate rx_cocaine, rx_amph, methylp, lisdex
        if non_tx:
            flags.add('STIM_MISUSE_NLP')
        else:
            flags.add('STIM_TX_NLP')
        
    if 'METHAMPHETAMINE' in cats:
        if non_tx:
            flags.add('ILLICIT_METHAMPHETAMINE_NLP')       
    
    if 'MDMA' in cats:
        if non_tx:
            flags.add('ILLICIT_MDMA_NLP')
        
    if 'UNSPECIFIED_STIMULANT' in cats:
        if non_tx:
            flags.add('STIM_NON_TX_UNSP_NLP')
        else:
            flags.add('STIM_TX_NLP')
        
    if 'ILLICIT_OPIOID' in cats:
        if non_tx:
            flags.add('OPIOID_ILLICIT_NLP')
        else:
            flags.add('OPIOID_ANY_NLP')            
        
    if 'RX_OPIOID' in cats:
        if non_tx:
            flags.add('OPIOID_MISUSE_NLP')
        else:
            flags.add('OPIOID_ANY_NLP')
        
    if 'UNSPECIFIED_OPIOID' in cats: 
        if non_tx:
            flags.add('OPIOID_NON_TX_UNSP_NLP')
        else:
            flags.add('OPIOID_ANY_NLP')   
                
    if 'FENTANYL' in cats:
        if non_tx:
            flags.add('OPIOID_ILLICIT_NLP') 
        else:
            flags.add('OPIOID_ANY_NLP')
        
    if 'MAT' in cats:
        if non_tx:
            flags.add('OPIOID_MISUSE_NLP')
        else:
            flags.add('OPIOID_NON_TX_UNSP_NLP')
            
    return(flags)

#This function is for free text. Determinations cannot initially be made at the row level but must be made individually
#for every mention within the row. Data table is exploded to make each mention its own row. Each row now can only have one flag

def determine_note_flag_freetext(cat, non_tx):        
        
    if cat == 'COCAINE':
        if non_tx:
            return('ILLICIT_COCAINE_NLP')
        else:
            return('')

    if cat == 'RX_STIM': #formerly separate rx_cocaine, rx_amph, methylp, lisdex
        if non_tx:
            return('STIM_MISUSE_NLP')
        else:
            return('STIM_TX_NLP')
        
    if cat == 'METHAMPHETAMINE':
        if non_tx:
            return('ILLICIT_METHAMPHETAMINE_NLP')  
        else:
            return('')
    
    if cat == 'MDMA':
        if non_tx:
            return('ILLICIT_MDMA_NLP')
        else:
            return('')
        
    if cat == 'UNSPECIFIED_STIMULANT':
        if non_tx:
            return('STIM_NON_TX_UNSP_NLP')
        else:
            return('')
        
    if cat == 'ILLICIT_OPIOID':
        if non_tx:
            return('OPIOID_ILLICIT_NLP')
        else:
            return('OPIOID_ANY_NLP')            
        
    if cat == 'RX_OPIOID':
        if non_tx:
            return('OPIOID_MISUSE_NLP')
        else:
            return('OPIOID_ANY_NLP')
        
    if cat == 'UNSPECIFIED_OPIOID': 
        if non_tx:
            return('OPIOID_NON_TX_UNSP_NLP')
        else:
            return('OPIOID_ANY_NLP')   
                
    if cat == 'FENTANYL':
        if non_tx:
            return('OPIOID_ILLICIT_NLP') 
        else:
            return('OPIOID_ANY_NLP')
        
    if cat == 'MAT':
        if non_tx:
            return('OPIOID_MISUSE_NLP')
        else:
            return('OPIOID_NON_TX_UNSP_NLP')
            
    #if you get to this point, there is some category that is unaccounted for
    if cat is None or cat == '':
        return ('')
    raise ValueError(f"This category is unaccounted for in category-to-flag mapping: {cat}")
    
#Load the model and tokenizer. Only needed for free text option
if text_format == "free text":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #loading from path or huggingface?
    if Path(model_indir).exists():
        model_source = 'local'
    else:
        model_source = 'hosted'
    
    if model_source == 'local':
        model = BertForSequenceClassification.from_pretrained(str(Path(model_indir)))      
        tokenizer = BertTokenizer.from_pretrained(str(Path(model_indir)))
        
    else:        
        model = BertForSequenceClassification.from_pretrained(model_indir)          
        tokenizer = BertTokenizer.from_pretrained(model_indir) 
        
    model.to(device)
    model.eval()
    max_len = 150

#output variables for this code defined
output_vars = ['STIM_TX_NLP', 'ILLICIT_COCAINE_NLP', 'ILLICIT_METHAMPHETAMINE_NLP',
'ILLICIT_MDMA_NLP', 'STIM_MISUSE_NLP', 
'STIM_NON_TX_UNSP_NLP', 'OPIOID_ANY_NLP', 'OPIOID_ILLICIT_NLP',
'OPIOID_MISUSE_NLP', 'OPIOID_NON_TX_UNSP_NLP', 'DRUGSCREEN_NLP']
stim_cats = set(['RX_STIM', 'UNSPECIFIED_STIMULANT'])
output_vars_df = pd.DataFrame({x: [0] * chunksize for x in output_vars})

#build the regular expressions used to determine drugscreens. They looked different in our FHIR data and free text data
#so we built them separately

screen_regex_free_text = re.compile(r"(\bndet\b)|(\^g=)|(urine\s[neg|pos])|(scr[n\s]+)|drug screen|tox screen|urine tox|drug abuse screen|drugs of abuse|drugs urin|screen urin|Ur negative|ur positive", re.I)

#screen_regex_ehr = re.compile(r"drug screen|drugs of abuse|drug abuse screen|drugs urine|drug abuse urine panel|screen urin", re.I)
lab_terms_df = search_terms['LAB_TERMS']
lab_terms_df.columns = [x.upper() for x in lab_terms_df.columns]
lab_query = Query(lab_terms_df, query_type = 'boundary with s',
                  input_type = "dataframe")
lab_regex = lab_query.build_re()

chunksize = 1000

if input_type.upper() == 'SQL':
    if sql_query is None or sql_query == '' or cnxn_string is None or cnxn_string == '':
        raise ValueError("Input type is SQL but missing either a connection string or a query string")
    cnxn = pyodbc.connect(cnxn_string)
    df_iter = pd.read_sql(sql_query, cnxn, chunksize = chunksize)
elif input_type.upper() == 'CSV':
    if infile_path is None or infile_path == '':
        raise ValueError("Input type is CSV but missing input file path")
    df_iter = pd.read_csv(Path(infile_path), iterator = True, chunksize = chunksize)
elif input_type.upper() == 'SAS':
    if infile_path is None or infile_path == '':
        raise ValueError("Input type is SAS but missing input file path")
    df_iter = pd.read_sas(Path(infile_path), encoding = 'latin-1', iterator = True, chunksize = chunksize)    
else:
    raise ValueError("Input type is not recognized")

if text_format == "FHIR":
    fhir_dfs = []
    for counter, df in enumerate(df_iter):
        print(f"Processing dataframe {counter}")
        df.columns = df.columns.str.upper()
        df = pd.concat([df, output_vars_df.head(len(df))], axis = 1)
        if len(df) == 0:    
            continue    
            
        df[col_to_search] = df[col_to_search].fillna('') 
        df['MATCHES'] = df.apply(lambda row: set([x.lower() for x in re.findall(drug_regex, row[col_to_search])]), axis = 1)
        

        df['DRUGSCREE N_NLP'] = np.where((df[col_to_search].str.contains(lab_regex, regex = True)) & \
                                            (df[note_type_col] == drugscreen_note_type) , 1, 0) 

                                                                                
        drugscreen_type_df = df.loc[df[note_type_col] == drugscreen_note_type]
        #the above is not eligible for further evaluation
        df = df.loc[df[note_type_col] != drugscreen_note_type]
        
        df['CATS'] = df.apply(lambda row: set([cat_d[x.lower()] for x in row['MATCHES']]), axis=1)
        df['NON_TX'] = np.where(df[note_type_col].isin(ehr_diag_titles), 1, 0)

        #where cats are rx stim cats and note title is medications
        df['STIM_TX_NLP'] = df.apply(lambda row: 1 if row[note_type_col] == meds_note_type and \
                                             row['CATS'].intersection(stim_cats) != set() else 0, axis=1)

        meds_type_df = df.loc[df[note_type_col] == meds_note_type]
        #the above is not eligible for further evaluation
        df = df.loc[df[note_type_col] != meds_note_type]
                                             
        #after determining screenings and tx use, filter out rows that are of no interest 
        #i.e. rows with no matches
        df = df.loc[df['MATCHES'] != set()]
        
        df['FLAGS'] = df.apply(lambda row: determine_note_flag_FHIR(row['CATS'], row['NON_TX']), axis=1)
    
        
        def update_flags(row):
            for flagname in row['FLAGS']:
                if flagname not in row.index:
                    raise ValueError(f"Flag {flagname} does not exist in the dataframe")
            row[flagname] = 1
            return(row)
            
        df = df.apply(update_flags, axis = 1)
        df = pd.concat([df, drugscreen_type_df, meds_type_df])
        df[output_vars] = df[output_vars].fillna(0)
        df = df[df[output_vars].max(axis = 1) > 0].copy()
        df = df[cols_to_keep + output_vars]
        if group_cols is not None and group_cols != '':
            df = df.groupby(group_cols, as_index = False).max()
        fhir_dfs.append(df)
        
if text_format == "free text":
    free_text_dfs = []
    for counter, df in enumerate(df_iter):
        #test
        print(f"Processing df {counter}")
        df.columns = df.columns.str.upper()
        df[col_to_search] = df[col_to_search].fillna('') 
        df['MATCHES'] = df.apply(lambda row: set([x.lower() for x in re.findall(drug_regex, row[col_to_search])]), axis = 1)
        df = pd.concat([df, output_vars_df.head(len(df))], axis = 1)

        #For free text, we don't even consider rows that have no drug terms of interest in them
        df = df[df['MATCHES'].apply(lambda x: len(x) > 0)].copy()

        
        window = 70
        #Extract snippet of text around each drug term from our list of drug terms.
        #Returns a list of 2-tuples, being the snippet of text and the drugterm in the snippet
        df['SNIPPETS'] = df.apply(lambda row: [extract_snippets(row[col_to_search], x, window) for x in \
                                                         re.finditer(drug_regex, row[col_to_search])], axis = 1)
        
        #Explode list of 2-tuples out so that there is only 1 2-tuple per cell
        df = df.explode('SNIPPETS').reset_index(drop=True)

        #Normalize text part of 2-tuple, returns (normalized_text, drugterm)
        df['NORMALIZED_SNIPPETS'] = df.apply(lambda row: (normalize_w_drugterm(row['SNIPPETS'][0], row['SNIPPETS'][1]),
                                                                    row['SNIPPETS'][1]), axis = 1)
         #Put normalized text and drug term in diffent columns
        df[['NORMALIZED_SNIPPETS', 'MATCH_TERM']] = df['NORMALIZED_SNIPPETS'].apply(lambda x: pd.Series([x[0], x[1]]))

        #Drug screens get their own flag, and aren't evaluated further for anything else
        df['DRUGSCREEN_NLP'] = np.where(df['NORMALIZED_SNIPPETS'].str.contains(screen_regex_free_text, regex=True), 1, 0)

        #Use score questionnaires are pretty standard and don't in and of themselves indicate much. exclude these 
        df['USE_SCORE'] = np.where(df['NORMALIZED_SNIPPETS'].str.contains(r"\bscore\b", case=False, regex=True), 1, 0)

        #Save these for later
        drugscreen_df = df[df['DRUGSCREEN_NLP'] == 1].copy()

        #Only rows that are not drugscreens and not use scores will be evaluated
        to_eval_df = df[(df['DRUGSCREEN_NLP'] != 1) & (df['USE_SCORE'] != 1)].copy()
        
        #Get the categories of each match term
        to_eval_df['CAT'] = to_eval_df.apply(lambda row: cat_d[row['MATCH_TERM'].lower()] , axis=1)

        #Get flags for overrides. These don't need to be evaluated by a model.
        to_eval_df['FLAG'] = ''       
        to_eval_df['FLAG'] = np.where(to_eval_df['CAT'] == 'OPIOID_MISUSE_OVERRIDE', 'OPIOID_MISUSE_NLP', to_eval_df['FLAG'])
        to_eval_df['FLAG'] = np.where(to_eval_df['CAT'] == 'STIMULANT_MISUSE_OVERRIDE', 'STIM_MISUSE_NLP', to_eval_df['FLAG'])
        to_eval_df['FLAG'] = np.where(to_eval_df['CAT'] == 'STIMULANT_NON_TX_UNSP_OVERRIDE', 'STIM_NON_TX_UNSP_NLP', to_eval_df['FLAG'])
        to_eval_df['FLAG'] = np.where(to_eval_df['CAT'] == 'OPIOID_NON_TX_UNSP_OVERRIDE', 'OPIOID_NON_TX_UNSP_NLP', to_eval_df['FLAG'])

        override_df = to_eval_df[to_eval_df['FLAG'] != ''].copy()
        to_eval_df = to_eval_df[to_eval_df['FLAG'] == ''].copy()

        #Pass to model only those texts the model needs to evaluate for non-therapeutic status
        normalized_as_batch = tokenize_and_encode(to_eval_df['NORMALIZED_SNIPPETS'].values.tolist())
        classifications = np.argmax(eval_with_label(normalized_as_batch), axis=1)
        to_eval_df['NON_TX'] = classifications

        #Now that we have a category and a non-therapeutic status, we can determine the flag
        to_eval_df['FLAG'] = to_eval_df.apply(lambda row: determine_note_flag_freetext(row['CAT'], row['NON_TX']), axis=1)
        #Remove rows with no flags. Everything that remains in FLAG column should equal one of the output variables
        to_eval_df = to_eval_df.loc[to_eval_df['FLAG'] != '']

        #For each value in FLAG, mark the column with that name as a 1. Add the override_df back in before doing this
        to_eval_df = pd.concat([to_eval_df, override_df])
        new_rows = []
        for _, row in to_eval_df.iterrows():
            flag = row['FLAG']
            row[flag] = 1
            new_row = row.copy()
            new_rows.append(new_row)
            
        to_eval_df = pd.concat(new_rows, axis=1).T

        #Now add back the drugscreens, which already have their one flag (DRUGSCREEN_NLP)
        df = pd.concat([to_eval_df, drugscreen_df])

        #Drop all columns that are not part of our output
        df = df[cols_to_keep + output_vars]

        #For space, group by here as well. We will have to do this once after searching is done, too.
        if group_cols is not None and group_cols != '':
            df = df.groupby(group_cols, as_index = False).max()

        #Again, for space, let's not keep any rows that don't have any flags
        df = df[df[output_vars].max(axis = 1) > 0].copy()
        free_text_dfs.append(df)
        
if text_format == "free text":
    all_dfs = pd.concat(free_text_dfs)
else:
    all_dfs = pd.concat(fhir_dfs)
    
if group_cols is not None and group_cols != '':
    all_dfs = all_dfs.groupby(group_cols, as_index = False).max()

all_dfs['STIM_ILLICIT_NLP'] = np.where((all_dfs['ILLICIT_MDMA_NLP'] ==1) | \
        (all_dfs['ILLICIT_COCAINE_NLP'] ==1) | \
        (all_dfs['ILLICIT_METHAMPHETAMINE_NLP']==1), 1, 0)


all_dfs['STIM_NON_TX_UNSP_NLP'] = np.where((all_dfs['STIM_MISUSE_NLP'] == 1) | \
        (all_dfs['STIM_ILLICIT_NLP']==1), 0, all_dfs['STIM_NON_TX_UNSP_NLP'])

all_dfs['OPIOID_NON_TX_UNSP_NLP'] = np.where((all_dfs['OPIOID_MISUSE_NLP']==1) | \
        (all_dfs['OPIOID_ILLICIT_NLP'] == 1), 0, all_dfs['OPIOID_NON_TX_UNSP_NLP'])

all_dfs['STIM_ANY_NON_TX_NLP'] = np.where((all_dfs['STIM_MISUSE_NLP'] == 1) | \
        (all_dfs['STIM_ILLICIT_NLP']==1) | (all_dfs['STIM_NON_TX_UNSP_NLP'] == 1),
        1, 0)

all_dfs['OPIOID_ANY_NON_TX_NLP'] = np.where((all_dfs['OPIOID_NON_TX_UNSP_NLP'] == 1) | \
        (all_dfs['OPIOID_MISUSE_NLP']==1) | (all_dfs['OPIOID_ILLICIT_NLP'] == 1),
        1, 0)

all_dfs['STIM_ANY_NLP'] = np.where( (all_dfs['STIM_ANY_NON_TX_NLP'] ==1) | \
        (all_dfs['STIM_TX_NLP']==1), 1, 0)

all_dfs['OPIOID_ANY_NLP'] = np.where(all_dfs['OPIOID_ANY_NON_TX_NLP'] ==1, 1, 
        all_dfs['OPIOID_ANY_NLP'])

#We didn't get enough test cases to test the accuracy of some categories. These are dropped here.
#The parent columns that they contribute to can remain, however.
cols_to_drop = ['STIM_NON_TX_UNSP_NLP', 'STIM_MISUSE_NLP', 'OPIOID_NON_TX_UNSP_NLP', 'OPIOID_MISUSE_NLP']
all_dfs.drop(columns = cols_to_drop, inplace=True)

all_dfs.to_csv(Path(results_path), index=False)


    
