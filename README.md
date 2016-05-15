# IRC

Information Retrieval C


## Environment

The application was built using python2.7, and packages specified under [requirements.txt](requirements.txt)

You can install them with:

```sh
pip install -r requirements.txt
```

## Usage

### Running Queries

```sh
Usage:  python -m irc.main --index [index] --q [term] --n [10]

Where:

    --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-S
    --q                 : Search query term
    --n                 : Number of matches to be returned. Default is 10, * for all

```

### Measuring Performance

Measures performance of an index using the MED.QRY and MED.REL queries and measures.

```sh
Usage: python -m irc.evaluation
```

The script will generate an image file with the evaluation of all 3 indeces available that can be compared:

  - TF: Term-frequency
  - TF-IDF: Term-frequency/Inverse document frequency
  - TF-IDF-Prob: Term-frequency/Probabilistic inverse document frequency
  
  
### Feedback

```sh
    Usage:  python -m irc.feedback --index [index] --q [ID] --n [default 10]

    Where:

        --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-Prob
        --q                 : Query: 1-30
        --n                 : Number of matches to be returned. Default is 10, * for all
``` 
        