# Guide to the NLP / Clinical Notes component <p> of the NCHS Stimulant and Opioid Use Algorithm

**Author: Nikki Adams**

**nadams2@cdc.gov**

**Edited: January 14, 2025**

## Background 

The Python code in this repository was developed as part of the project titled “Utilizing Natural Language Processing and Machine Learning to Enhance the Identification of Stimulant and Opioid-Involved Health Outcomes in the National Hospital Care Survey, supported by the Office of the Secretary – Patient-Centered Outcomes Research Trust Fund (OS-PCORTF) in the 2023 fiscal year. Its primary aim is to detect non-therapeutic use (that is, use of illicit drugs or prescription misuse) of stimulants and opioids in clinical note text. It was originally developed for the National Hospital Care Survey (NHCS), an establishment survey of non-federal, non-institutional hospitals with at least six staffed inpatient beds conducted by the National Center for Health Statistics (NCHS). This work was initially developed to analyze data from the 2020 survey year. The code presented here has been adapted for general use by the public.

<ins> Links to other repositories: </ins>

There are two medical code-based components to this same project, and they  can be found here:

* []	Stimulant and Opioid Use:
  * CDCgov/stimulant_opioid_algorithm_medical_codes_R
  * CDCgov/stimulant_opioid_algorithm_medical_codes_SAS

<ins>Related repositories:</ins>

The stimulant algorithm is the third in a series of related substance-use-related algorithms. For your reference, these algorithms can be found in the following repositories.

  * [ ] Algorithm to detect opioid use, selected mental health issues, and substance use disorders in medical codes: 
    * https://github.com/CDCgov/Opioid_SUD_MHI_MedCodes

  * [ ] Algorithms to detect opioid use, selected mental health issues, and substance use disorders in clinical notes: 
    * https://github.com/CDCgov/Opioid_Involvement_NLP
    * https://github.com/CDCgov/SUD_MHI_NLP

<ins> Methodology and Data Dictionary </ins>

A methodology report is under preparation and will be available at the below address when published:

* https://www.cdc.gov/nchs/products/series/series02.htm
  
The results of applying the algorithm to the NHCS 2020 data will be available in the NHCS Research Data Center, and the data dictionary describing the output variables of this algorithm will be available on the NHCS website:

* https://www.cdc.gov/nchs/nhcs/data/index.html

## Code Overview

This repository includes code, sample input data, and a sample configuration file for the user to specify select settings. Code is provided as a Notebook and a Python-interpretable (.py) file. The two code files perform the same action; a user only needs to use one.

<ins> Requirements </ins>

A requirements file is included for all necessary packages. These can be installed at a command line, when in the same directory as the requirements file, with one of the following two commands, depending on your environment and Python installation:

```
pip install -r requirements.txt
```
or

```
py -m pip install -r requirements.txt
```

Alternatively, these can be installed in the Notebook individually, such as:
```
!pip install pyodbc
```
### Configuration File 

The configuration file should not require the user to alter the code files (.py and .ipynb). It should be placed in the same directory as the code, and when the code is run, it will read in all necessary configurations from that file. Options should not be placed in quotation marks. Specifications are case-sensitive, so be sure to specify values in their proper case (as with note type values, for example).

Options under [INPUT_SETTINGS]:

* input_type: Required.
  
  Options are:
  
    * CSV  
    * SQL
    * SAS
    
* text_format: Required.
  
  Options are:
  
    * FHIR  
    * free text
    
* model_indir: Required if text_format is free text. This points to the model in the example configuration file and should not need to be changed.
  
* cnxn_string: Required if input_type is SQL. This is the connection string used to connect to the database that stores your data.
  
* sql_query: Required if input_type is SQL. This is the query that will be passed to the SQL Server identified in the cnxn_string argument.
  
* infile_path: Required if input_type is SAS or CSV.
  
* search_terms_path: Required. This should not be changed.

Options under [SEARCH_SETTINGS]:

* col_to_search: Required. Regardless of input_type, this is the name of the column that contains the clinical note text to search.
  
* note_type_col: Required if text_format is FHIR. This is the name of the column that contains the metadata about what type of note this observation is.
  
* drugscreen_note_type: Required if text_format is FHIR. This is the value in the note_type_col that corresponds to results of drug screenings. Only one is allowed.
  
* meds_note_type: Required if text_format is FHIR. This is the value in the note_type_col that corresponds to prescribed or ordered medications. Only one is allowed.
  
* ehr_diag_titles: Required if text_format is FHIR. These are the values in the note_type_col that correspond to note types where diagnoses are likely to be found. More than one is allowed. They should be separated by commas.

Options under [OUTPUT_SETTINGS]:

* results_path: Required. This is the file to which the results will be written. Results are output as a CSV file.
  
* cols_to_keep: Required. These are the columns to keep for each row of results. These would be columns such as a unique identifier or linkage keys. More than one is allowed. They should be separated by commas. Also, these are the columns to which results should be aggregated. For example, if the data has multiple events (as rows) for each encounter, but results are desired at the encounter level, then the column indicating the encounter ID number would be the column to aggregate to. Columns in the output file that are not part of cols_to_keep will have a max function applied to them, so if using this option, it is best that all columns that are in that are not in cols_to_keep are numeric columns.

### Note about free text and FHIR options:

The free text option expects unstructured text and evaluates each drug mention for its therapeutic or non-therapeutic status individually. It does not rely on the presence of any metadata, such as note types.

The FHIR option expects semi-structured text in the XML-based FHIR standard (https://build.fhir.org). In particular, this option requires a separate column of the input data that designates the note type and the values in that column that indicate medications, laboratory results, and diagnostic-type note types. Please see the included configuration file for an example. Note that these values can be whatever is relevant to the data content. In the example, the note types are descriptive strings, but the FHIR OID values (https://build.fhir.org/oids.html) would work just as well if that is how note types are designated in the data. This FHIR option evaluates the non-therapeutic status of all drug mentions within a single note type at once.

### Running the code

After configuration specifications are all set, the code can be run with any of the following methods:

* (1)	At a command line with:
```
python Stimulant_Opioid_Non_Therapeutic_Use.py
```

* (2)	In an IDE (such as Spyder or Pycharm) by running the file:
```
Stimulant_Opioid_Non_Therapeutic_Use.py
```

* (3)	In a notebook with “Run All Cells”



## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
The repository utilizes code licensed under the terms of the Apache Software
License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](DISCLAIMER.md)
and [Code of Conduct](code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records, but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
