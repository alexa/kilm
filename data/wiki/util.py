from contextlib import contextmanager
import mwparserfromhell
import re
import signal
import spacy
from wikimapper import WikiMapper
import xml.sax

generic_title_start = ["Category:", "File:", "Template:", "Draft:", "Wikipedia:"]


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class WikiTitleIdResolver:
    def __init__(self, wiki_db):
        """
        This class uses the WikiMapper python package
        :param wiki_db: Address to the Wikipedia index. Follow the instructions at
            https://github.com/jcklie/wikimapper#create-your-own-index to create this
            index
        """
        self.mapper = WikiMapper(wiki_db)

    def title2id(self, title):
        """
        Give a Wikipedia page title return the ID
        """
        return self.mapper.title_to_id(title)

    def id2title(self, id):
        """
        Given the ID of a Wikipedia page return the page title
        """
        return self.mapper.id_to_title(id)

    @staticmethod
    def surface2title(surface_form):
        """
        Turn surface form to page title. For instance in

            "marked the end of the [[classical era of anarchism]]. In the last"

        turn "classical era of anarchism" to "Classical_era_of_anarchism". Rules
        of this function come from two places:

        1) https://en.wikipedia.org/wiki/Help:Link where it says:
            "The link target is case-sensitive except for the first character
            (so [[atom]] links to "Atom" but [[ATom]] does not)."

        2) https://en.wikipedia.org/wiki/Wikipedia:Page_name where it says:
            "some translation occurs, such as spaces are replaced with underscores"

        :param surface_form: Surface form of a Wikipedia link
        :return: Page title associated with the surface form
        """
        surface_form = surface_form.capitalize()
        return "_".join(surface_form.split())


class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""

    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self.pages = []
        self.cleaner = Cleaner()
        self.nlp = spacy.load('en_core_web_sm')

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            self.pages.append((self._values['title'], self._values['text']))


class WikiPageProcessor:
    def __init__(self):
        self.cleaner = Cleaner()
        self.nlp = spacy.load('en_core_web_sm')

    def process(self, page):
        """
        Process the text of a wikipedia page

        returns: A dictionary that contains a title, short description, first sentence, and text of the article
        """
        title = page[0]
        text = self.cleaner.clean_text(page[1].strip())
        text = self.text_cleanup(text)

        short_desc = self.get_short_desc(page[1])
        short_desc_clean, _ = self.cleaner.remove_links(short_desc)

        parsed_wikicode = mwparserfromhell.parse(page[1])
        parsed_wiki = parsed_wikicode.strip_code().strip()
        first_parag = self.text_cleanup(parsed_wiki).split("\n")[0]

        try:
            first_sent = str([i for i in self.nlp(first_parag).sents][0])
        except:
            print(title, first_parag)
            first_sent = ""
        first_sent_clean, _ = self.cleaner.remove_links(first_sent)

        return {
            'title': title,
            'short_description': short_desc_clean,
            'first_sentence': first_sent_clean,
            'text': text
        }

    @staticmethod
    def get_short_desc(text):
        start = '{{short description|'
        end = '}}'
        start_ind = text.lower().find(start)
        end_ind = -1
        if start_ind > -1:
            end_ind = text[start_ind:].find(end)
            if end_ind > -1:
                end_ind += start_ind

        if start_ind > -1 and end_ind > -1:
            return text[start_ind + len(start):end_ind]
        else:
            return ""

    def text_cleanup(self, text):
        text = self.remove_tags(text)
        text = self.remove_unwanted_sections(text)
        text = self.remove_hyperlinks(text)
        text = text.strip()
        return text

    @staticmethod
    def remove_unwanted_sections(text):
        exclude_sections = ["reference", "references", "see also", "sources",
                            "citations", "secondary sources", "external links", "primary sources",
                            "tertiary sources", "further reading", "notes", "footnote", "footnotes"]
        content = []
        skip = False
        for l in text.splitlines():
            line = l.strip()
            for ex in exclude_sections:
                pat = re.compile("=+\s*" + ex + "\s*=+")
                if re.match(pat, line.lower()):
                    skip = True  # replace with break if this is the last section
                    continue
            if skip:
                continue
            content.append(line)

        return '\n'.join(content) + '\n'

    @staticmethod
    def remove_hyperlinks(text):
        """
        Remove the patter [[abc:xyz]]
        """
        pat = "\[\[(.*?):(.*?)]]"
        text = re.sub(pat, '', str(text))
        return text

    @staticmethod
    def remove_tags(text):
        """
        Remove everything inside {} and <>
        """
        pat_lt_gt = re.compile("&lt;\s*(.*?)\s*&gt;")
        text = re.sub(pat_lt_gt, "", text)

        out = ''
        skip1c = 0
        skip2c = 0
        i = 0
        while i < len(text):
            if text[i] == '{':  # and text[i+1] == '{':
                skip1c += 1
                i += 1
            elif text[i] == '<':
                skip2c += 1
                i += 1
            elif text[i] == '}' and skip1c > 0:  # and text[i+1] == '}' :
                skip1c -= 1
                i += 1
            elif text[i] == '>' and skip2c > 0:
                skip2c -= 1
                i += 1
            elif skip1c == 0 and skip2c == 0:
                out += text[i]
                i += 1
            else:
                i += 1

        # Remove "[http:xyz.com link to the address]"
        link_pat = re.compile("\[\s*http(.*?):(.*?)]")
        out = re.sub(link_pat, "", out)
        return out


class Cleaner(object):

    def __init__(self):
        pass

    def clean_text(self, text):

        text = self._remove_file_links(text)
        text = self._remove_image_links(text)
        # text = self._remove_external_links(text)
        text = self._remove_refs(text)
        text = self._remove_emphasises(text)
        text = self._remove_comments(text)
        text = self._remove_langs(text)
        # text = self._remove_titles(text)
        text = self._remove_choices(text)
        text = self._remove_templates(text)
        text = self._remove_htmls(text)
        text = self._remove_lists(text)
        text = self._remove_indents(text)
        text = self._remove_styles(text)
        text = self._remove_spaces(text)
        text = self._remove_continuous_newlines(text)
        return text.strip()

    def _remove_file_links(self, text):
        """Remove links like `[[File:*]]`"""
        text = self._remove_resource_links(text, 'File')
        text = re.sub(r'^File:.*$', '', text, flags=re.MULTILINE)
        return text

    def _remove_image_links(self, text):
        """Remove links like `[[File:*]]`"""
        return self._remove_resource_links(text, 'Image')

    @staticmethod
    def _remove_resource_links(text, resource):
        """Remove links likes `[[*:*]]`"""
        pattern = '[[' + resource + ':'
        pattern_begin = text.find(pattern)
        if pattern_begin == -1:
            return text
        begin, removed = 0, ''
        while begin < len(text):
            if pattern_begin > begin:
                removed += text[begin:pattern_begin]
            pattern_end, depth = pattern_begin + 2, 2
            while pattern_end < len(text):
                ch = text[pattern_end]
                pattern_end += 1
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        break
            if depth == 0:
                begin = pattern_end
            else:
                removed += text[begin]
                begin += 1
            pattern_begin = text.find(pattern, begin)
            if pattern_begin == -1:
                break
        if len(text) > begin:
            removed += text[begin:]
        return removed

    @staticmethod
    def _remove_external_links(text):
        """Remove links like [*]"""
        return re.sub(r'\[h[^ ]+ (.*?)\]', r'\1', text)

    @staticmethod
    def _remove_refs(text):
        """Remove patterns like <ref*>*</ref>"""
        text = re.sub(r'<ref[^/]*?/>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<ref.*?</ref>', '', text, flags=re.IGNORECASE | re.DOTALL)
        # text = re.sub(r'{{Refbegin.*?Refend}}', '', text, flags=re.IGNORECASE | re.DOTALL)
        return text

    @staticmethod
    def _remove_emphasises(text):
        """Remove patterns like '''*'''"""
        text = re.sub(r"'''(.*?)'''", r'\1', text, flags=re.DOTALL)
        text = re.sub(r"''(.*?)''", r'\1', text, flags=re.DOTALL)
        return text

    @staticmethod
    def _remove_comments(text):
        """Remove patterns like <!--*-->"""
        return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    @staticmethod
    def _remove_langs(text):
        """Remove patterns like {{lang-*|*}}}"""
        return re.sub(r'{{lang(-|\|).*?\|(.*?)}}', r'\2', text, flags=re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _remove_titles(text):
        """Remove patterns like ==*=="""
        return re.sub(r'(={2,6})\s*(.*?)\s*\1', r'\2', text)

    @staticmethod
    def _remove_choices(text):
        """Remove patterns like -{zh-hans:*; zh-hant:*}-"""
        text = re.sub(
            r'-{.{,100}?zh(-hans|-cn|-hk|):(.{,100}?)(;.{,100}?}-|}-)',
            r'\2', text,
            flags=re.DOTALL
        )
        text = re.sub(r'-{.{,100}?:(.{,100}?)(;.{,100}?}-|}-)', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'-{(.{,100}?)}-', r'\1', text, flags=re.DOTALL)
        return text

    @staticmethod
    def _remove_templates(text):
        """Remove patterns like {{*}}"""
        begin, removed = 0, ''
        while begin < len(text):
            pattern_begin = text.find('{{', begin)
            if pattern_begin == -1:
                if begin == 0:
                    removed = text
                else:
                    removed += text[begin:]
                break
            if pattern_begin > begin:
                removed += text[begin:pattern_begin]
            pattern_end, depth = pattern_begin + 2, 2
            while pattern_end < len(text):
                ch = text[pattern_end]
                pattern_end += 1
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        link = text[pattern_begin + 2:pattern_end - 2]
                        parts = link.split('|')
                        template_type = parts[0].split(' ')[0].lower()
                        if len(parts) == 1:
                            if all(map(lambda x: x in {'"', "'", ' '}, parts[0][:])):
                                removed += parts[0].replace(' ', '')
                        elif len(parts) in [2, 3]:
                            if template_type in {'le'} or template_type.startswith('link-'):
                                removed += parts[1]
                        break
            begin = pattern_end
        return removed

    @staticmethod
    def _remove_htmls(text):
        return re.sub(r'<(.*?)>', '', text, flags=re.DOTALL)

    @staticmethod
    def _remove_lists(text):
        return re.sub(r'^\s*[\*#]\s*', '', text, flags=re.MULTILINE)

    @staticmethod
    def _remove_indents(text):
        return re.sub(r'^\s*[:;]\s*', '', text, flags=re.MULTILINE)

    @staticmethod
    def _remove_styles(text):
        return re.sub(r':?{\| (style|class)=.*?\|}', '', text, flags=re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _remove_spaces(text):
        return text.replace('\u200b', '')

    @staticmethod
    def _remove_continuous_newlines(text):
        return re.sub(r'\n{2,}', '\n', text)

    @staticmethod
    def remove_links(text):
        """
        Receives a wikipedia text and removes the hyperlinks from the text. These
        hyperlinks appear in between [[ and ]].
        Args:
            text: Some wikipedia text

        Returns:
            1) text without hyperlinks
            2) a list of removed links with their detailed information (URL, place within input text, ...)
        """
        begin, hl_removed_text, hyperlinks = 0, '', []   # hl_removed_text: text with hyperlinks removed
        while begin < len(text):
            pattern_begin = text.find('[[', begin)
            if pattern_begin == -1:
                if begin == 0:
                    hl_removed_text = text
                else:
                    hl_removed_text += text[begin:]
                break
            if pattern_begin > begin:
                hl_removed_text += text[begin:pattern_begin]
            pattern_end, depth = pattern_begin + 2, 2
            while pattern_end < len(text):
                ch = text[pattern_end]
                pattern_end += 1
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        hyperlink = text[pattern_begin + 2:pattern_end - 2]
                        parts = hyperlink.split('|')
                        if len(parts) == 1:
                            if ':' in hyperlink:
                                pure = hyperlink.split(':')[-1]
                                hyperlinks.append({
                                    'begin': len(hl_removed_text),
                                    'end': len(hl_removed_text) + len(pure),
                                    'hyperlink': hyperlink,
                                    'text': pure
                                })
                                hl_removed_text += pure
                            else:
                                hyperlinks.append({
                                    'begin': len(hl_removed_text),
                                    'end': len(hl_removed_text) + len(text),
                                    'hyperlink': hyperlink,
                                    'text': hyperlink
                                })
                                hl_removed_text += hyperlink
                        elif len(parts) == 2:
                            hyperlinks.append({
                                'begin': len(hl_removed_text),
                                'end': len(hl_removed_text) + len(parts[1]),
                                'hyperlink': parts[0],
                                'text': parts[1]
                            })
                            hl_removed_text += parts[1]
                        # Stop scanning the rest of the sentence for finding the end of hyperlink (`]]`)
                        break
            begin = pattern_end
        return hl_removed_text.strip(), hyperlinks
