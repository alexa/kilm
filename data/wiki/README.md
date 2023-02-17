# Overview


This directory contains code that goes through the dump of Wikipedia and
extracts a short description for each article. This data shall be used
for the Knowledge Injection into Language Model (KILM) approach.

Many Wikipedia articles already have a "short_description":

    "Android (robot)": "Robot resembling a human"
    "Alberta": "Province of Canada"
    "Affirming the consequent": "type of fallacious argument (logical fallacy)"


Many others don't for which we take the first sentence of the text of the article
as a short description:

    "Algorithms (journal)": "Algorithms is a monthly peer-reviewed open-access
        scientific journal of mathematics, covering design, analysis, and
        experiments on algorithms."

    "Argument (disambiguation)": "In logic and philosophy, an argument is an
        attempt to persuade someone of something, or give evidence or reasons
        for accepting a particular conclusion."

This code provides scripts to process wikipedia articles in order
to get the short description of each article. Plus it cleans up the
text (remove XML tags and remove unwanted sections such as references) 
of wikipedia such that the hyperlinks are maintained as they are 
needed for KILM.

# Outputs


Once the code is finished running two new files will be created:

1) `articles.json`: each line of this file is a dictionary with 4 keys:

    * `title`: name of the entity
    * `short_description`: short description for the entity
    * `first_sentence`: first sentence of the article
    * `text`: cleaned up text of the article
  
2) `short_desc.json`
	     Each line of this file is a json key value pair where the key is a wikipedia
	     article title and the value is a short description of that entity. Example:
   
	     {"Anarchism": "Political philosophy and movement"}
	     {"Autism": "Neurodevelopmental disorder involving social communication difficulties and repetitive behavior"}

Note that the `short_description` key in lines of `articles.json`
are identical to the values of rows in `short_desc.json`. In other words
all the information in `short_desc.json` exist in `article.json`, 
and the reason why we provide `short_desc.json` is ease of access
to a lookup table for short descriptions.

# Running the code


It requires a .xml.bz2 dump of wikipedia. For quick testing purposes you can download a much smaller
dump (only the first 100k lines of the xml file of the full dump) from [here](https://drive.corp.amazon.com/documents/mahdinam@/data/kelm_review/en-wiki-head-100k.xml.bz2
) [TODO: put this object in a public link and update the link for open source release].

After installing the requirements (specified in `requirements.txt`) for testing purposes the 
code could be run as follows:

`python wiki_short_desc.py --wiki_dump <address to the downloaded dump of wikipedia> --max_articles 5 --out_dir <a directory for output>`

The full dump of wikipedia in xml.bz2 format could be downloaded
from [here](https://dumps.wikimedia.org/enwiki/latest/). To run
the code on the entire dump (takes a day or two) the following command
should be run 

`python wiki_short_desc.py --wiki_dump <address to the downloaded dump of wikipedia> --out_dir <a directory for output>`

Note that in the above command `max_articles` is not set, and as 
a result, the processing happens for all Wikipedia articles. 