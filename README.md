# IRC

Information Retrieval C


## Usage

### Running Queries

```sh
Usage:  python2.7 -m irc.main --index [index] --q [term] --n [10]

Where:

    --index             : Index type. Options: Binary, TF, TF-IDF, TF-IDF-S
    --q                 : Search query term
    --n                 : Number of matches to be returned. Default is 10, * for all

```

### Measuring Performance

Measures performance of an index using the MED.QRY and MED.REL queries and measures.

```
python2.7 -m irc.server
```
