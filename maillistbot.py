#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import math
import time
import json
import email
import logging
import operator
import itertools
import collections
import configparser

import bs4
import requests
from segtok.segmenter import split_multi
from segtok.tokenizer import word_tokenizer, split_contractions

import porter
from imapidle import imaplib

logging.basicConfig(stream=sys.stderr, format='%(asctime)s [%(name)s:%(levelname)s] %(message)s', level=logging.DEBUG if sys.argv[-1] == '-v' else logging.INFO)

HSession = requests.Session()

decode_header = lambda s: str(email.header.make_header(email.header.decode_header(s)))

resourcepath = lambda *res: os.path.normpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__), *res))

re_greetings = re.compile('^(Hi|Dear|To whom|Hello).*,$', re.I)
re_endings = re.compile('(Yours|Best|Thank|Regards|Sincerely|Cheers).*,$', re.I)
re_quotehead = re.compile('\d+:\d+.*[,， ].+[,:：]$', re.I)
re_word = re.compile('\w+')

class BotAPIFailed(Exception):
    def __init__(self, ret):
        self.ret = ret
        self.description = ret['description']
        self.error_code = ret['error_code']
        self.parameters = ret.get('parameters')

    def __repr__(self):
        return 'BotAPIFailed(%r)' % self.ret

class TelegramBotClient:
    def __init__(self, apitoken):
        self.token = apitoken

    def bot_api(self, method, **params):
        for att in range(3):
            try:
                req = HSession.post(('https://api.telegram.org/bot%s/' %
                                    self.token) + method, data=params, timeout=45)
                retjson = req.content
                ret = json.loads(retjson.decode('utf-8'))
                break
            except Exception as ex:
                if att < 1:
                    time.sleep((att + 1) * 2)
                else:
                    raise ex
        if not ret['ok']:
            raise BotAPIFailed(ret)
        return ret['result']

    def __getattr__(self, name):
        return lambda **kwargs: self.bot_api(name, **kwargs)

def html2text(html):
    soup = bs4.BeautifulSoup(html, 'html5lib')
    [bq.decompose() for bq in soup.find_all('blockquote')]
    return ' '.join(soup.stripped_strings)

class IMAPClient:
    def __init__(self, config):
        self.config = config
        self.mail = None
        self.connect()
        self.lists = set(s.strip() for s in config['lists'].split(','))
        self.processed = set()

    def connect(self):
        if self.mail:
            self.mail.close()
            self.mail.logout()
        self.mail = imaplib.IMAP4_SSL(self.config['server'])
        self.mail.login(self.config['username'], self.config['password'])
        self.mail.select(self.config['folder'], readonly=True)
        logging.info('Connected.')

    def process_mail(self, mail):
        message = email.message_from_bytes(mail)
        tos = message.get_all('To', [])
        ccs = message.get_all('Cc', [])
        all_recipients = email.utils.getaddresses(tos + ccs)
        if all(r[1] not in self.lists for r in all_recipients):
            return
        return (
            decode_header(message['Subject']),
            message['From'],
            self.get_email_content(message)
        )

    def parse_fetch(self, results):
        for item in results:
            if isinstance(item, tuple):
                return item[1]

    def update(self):
        result, data = self.mail.uid('search', None, '(UNSEEN)')
        texts = []
        for uid in data[0].split():
            if uid in self.processed:
                continue
            self.processed.add(uid)
            result, data = self.mail.uid('fetch', uid, '(RFC822)')
            texts.append(self.process_mail(self.parse_fetch(data)))
        return texts

    def update_idle(self):
        try:
            uid, msg = next(self.mail.idle())
            self.mail.done()
            if uid in self.processed:
                return
            self.processed.add(uid)
            result, data = self.mail.uid('fetch', uid, '(RFC822)')
            return self.process_mail(self.parse_fetch(data))
        except StopIteration:
            return False
            logging.info('Disconnected from IDLE.')

    def get_email_content(self, message):
        maintype = message.get_content_maintype()
        textcontent = htmlcontent = None
        if maintype == 'multipart':
            for part in message.get_payload():
                if part.get_content_maintype() != 'text':
                    continue
                subtype = part.get_content_subtype()
                if subtype == 'plain':
                    raw = part.get_payload(decode=True)
                    textcontent = raw.decode(part.get_content_charset('utf-8'), errors='ignore')
                elif subtype == 'html':
                    raw = part.get_payload(decode=True)
                    htmlcontent = raw.decode(part.get_content_charset('utf-8'), errors='ignore')
        elif maintype == 'text':
            subtype = message.get_content_subtype()
            if subtype == 'plain':
                raw = message.get_payload(decode=True)
                textcontent = raw.decode(message.get_content_charset('utf-8'), errors='ignore')
            elif subtype == 'html':
                raw = message.get_payload(decode=True)
                htmlcontent = raw.decode(message.get_content_charset('utf-8'), errors='ignore')
        if textcontent:
            return textcontent
        elif htmlcontent:
            return html2text(htmlcontent)

def remove_quotes(text):
    out = []
    quoted = False
    for ln in text.splitlines():
        l = ln.strip()
        if not l:
            quoted = False
        elif l.startswith('>'):
            quoted = bool(l.lstrip('> '))
            continue
        elif quoted:
            continue
        elif re_greetings.match(l) or re_quotehead.search(l):
            continue
        elif re_endings.search(l) or l == '--':
            break
        out.append(l)
    return '\n'.join(out).strip()

def tokenize_document(text):
    return tuple(filter(operator.itemgetter(1), (
        Sentence(s.replace('\n', ' '),
        tuple(w for w in split_contractions(word_tokenizer(s))
        if re_word.search(w))) for s in split_multi(text))))

Sentence = collections.namedtuple(
    "Sentence", ("sentence", "words",))

SentenceInfo = collections.namedtuple(
    "SentenceInfo", ("sentence", "order", "rating",))

class TextRankSummarizer:
    """
    From sumy <https://github.com/miso-belica/sumy>.
    Source: https://github.com/adamfabish/Reduction
    """
    def __init__(self, stop_words=()):
        self._stemmer = porter.stem
        self._stop_words = frozenset(map(self.normalize_word, stop_words))

    def stem_word(self, word):
        return self._stemmer(self.normalize_word(word))

    def normalize_word(self, word):
        return word.lower()

    def _get_best_sentences(self, sentences, count, rating, *args, **kwargs):
        rate = rating
        if isinstance(rating, dict):
            assert not args and not kwargs
            rate = lambda s: rating[s]

        infos = (SentenceInfo(s, o, rate(s, *args, **kwargs))
            for o, s in enumerate(sentences))

        # sort sentences by rating in descending order
        infos = sorted(infos, key=operator.attrgetter("rating"), reverse=True)
        # get `count` first best rated sentences
        infos = infos[:count]
        # sort sentences by their order in document
        infos = sorted(infos, key=operator.attrgetter("order"))

        return tuple(i.sentence for i in infos)

    def __call__(self, document, sentences_count):
        '''
        `document` should be [sentences[words]]
        '''
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document, sentences_count, ratings)

    def rate_sentences(self, document):
        sentences_words = [(s, self._to_words_set(s)) for s in document]
        ratings = collections.defaultdict(float)

        for (sentence1, words1), (sentence2, words2) in itertools.combinations(
            sentences_words, 2):
            rank = self._rate_sentences_edge(words1, words2)
            ratings[sentence1] += rank
            ratings[sentence2] += rank

        return ratings

    def _to_words_set(self, sentence):
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _rate_sentences_edge(self, words1, words2):
        rank = 0
        for w1 in words1:
            for w2 in words2:
                rank += int(w1 == w2)

        if rank == 0:
            return 0.0

        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        return 0.0 if norm == 0.0 else rank / norm

def load_stop_words(filename):
    sw = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for ln in f:
            l = ln.strip()
            if l:
                sw.add(l)
    return sw

summarizer = TextRankSummarizer(load_stop_words(resourcepath('stop_english.txt')))

def send_message_for_mail(bot, mail, chat_id):
    if not mail:
        return
    subject, from_, content = mail
    if subject.startswith('Re: '):
        subject = subject[4:]
    from_name = email.utils.parseaddr(from_)[0]
    summary = ' '.join(s.sentence for s in
        summarizer(tokenize_document(remove_quotes(content)), 1))
    logging.info('Message from %s: %s (%s)' % (from_name, subject, summary))
    bot.sendMessage(
        chat_id=chat_id, text='%s: %s\n%s' % (from_name, subject, summary))

def load_config(filename):
    cp = configparser.ConfigParser()
    cp.read(filename)
    return cp

def main():
    config = load_config('config.ini')
    imapcli = IMAPClient(config['IMAP'])
    botcli = TelegramBotClient(config['Bot']['apitoken'])
    logging.info('Satellite launched.')
    for mail in imapcli.update():
        send_message_for_mail(botcli, mail, config['Bot']['chat_id'])
    while 1:
        mail = imapcli.update_idle()
        if mail is False:
            imapcli.connect()
            continue
        elif mail:
            send_message_for_mail(botcli, mail, config['Bot']['chat_id'])

if __name__ == '__main__':
    main()
