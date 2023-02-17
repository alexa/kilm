import argparse
from util import Cleaner, WikiPageProcessor, WikiXmlHandler, TimeoutException, time_limit, generic_title_start
import json
import os
import subprocess
import xml.sax


class WikiXmlShortDescRetriever:
    def __init__(self, wiki_xml_file, max_articles, output_dir):
        self.wiki_xml_file = wiki_xml_file
        self.max_articles = max_articles
        self.output_dir = output_dir

    def run(self):
        """
        Run the process function on articles of a wikipedia dump
        """
        # Object for handling xml
        handler = WikiXmlHandler()

        # Object for processing wikipedia text
        wiki_page_processor = WikiPageProcessor()

        # Parsing object
        parser = xml.sax.make_parser()
        parser.setContentHandler(handler)

        num_processed_pages = 0
        short_desc_dico = {}

        out_records = open(os.path.join(self.output_dir, 'articles.json'), "w")
        out_short_desc = open(os.path.join(self.output_dir, 'short_desc.json'), "w")

        last_shown_num_processed_pages = 0

        # Iteratively process wikipedia bz2 dump file and get a tuple of (title, text). This "text" is raw
        for line in subprocess.Popen(
                ['bzcat'],
                stdin=open(self.wiki_xml_file),
                stdout=subprocess.PIPE).stdout:

            parser.feed(line)
            # If the last element added is redirect, drop it
            if handler.pages:
                if len(handler.pages) > num_processed_pages:

                    if handler.pages[-1][1].startswith("#REDIRECT"):
                        handler.pages.pop()
                    else:
                        try:
                            with time_limit(30):
                                record = wiki_page_processor.process(handler.pages[-1])

                                # Filter out titles that start with "Template:", "File:", ...
                                if not list(filter(record['title'].startswith, generic_title_start)):
                                    json.dump(record, out_records)
                                    out_records.write('\n')
                                    short_desc = record['short_description'] if record['short_description'] \
                                        else record['first_sentence']

                                    short_desc_dico[record['title']] = short_desc

                                    sd_record = {record['title']: short_desc}
                                    json.dump(sd_record, out_short_desc)
                                    out_short_desc.write('\n')
                        except TimeoutException:
                            print("Timed out!", handler.pages[-1][0])

                num_processed_pages = len(handler.pages)
                # Stop when max_articles articles have been found, when max_articles is set to a positive value
                if 0 < self.max_articles < num_processed_pages:
                    break

                if num_processed_pages % 1000 == 0 and num_processed_pages > last_shown_num_processed_pages:
                    print("   Processed", num_processed_pages, "records")
                    last_shown_num_processed_pages = num_processed_pages

        out_records.close()
        out_short_desc.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--wiki_dump",
        default=None,
        type=str,
        required=True,
        help="Full path to the XML dump of Wikipedia with bz2 compression")
    argparser.add_argument(
        "--max_articles",
        default=-1,
        type=int,
        required=True,
        help="Maximum number of Wikipedia articles to process")
    argparser.add_argument(
        "--out_dir",
        default=False,
        type=str,
        help="Directory where the output should be saved"
    )
    args = argparser.parse_args()

    obj = WikiXmlShortDescRetriever(args.wiki_dump, args.max_articles, args.out_dir)

    obj.run()
