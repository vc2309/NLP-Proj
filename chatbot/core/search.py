#!/usr/local/bin/python3
# Search Handler of the Chatbot Webhook APP
# Created By Intralogue

import re
import urllib

from apiclient.discovery import build
import requests

from chatbot.helper import debug


class Search:
    def __init__(self, msg, context, query):
        self._DEVELOPER_KEY = "AIzaSyAnlk2LuHMdmISsnJ6-5UFRpno685dwKgg"
        self._GOOGLE_BOOK_API_SERVICE_NAME = "books"
        self._GOOGLE_BOOK_API_VERSION = "v1"

        self._YOUTUBE_API_SERVICE_NAME = "youtube"
        self._YOUTUBE_API_VERSION = "v3"

        # 1. find which type of content user want to search
        video_context = re.findall(
            r'video(s)?', context, re.I
        )
        video_msg = re.findall(
            r'(video(s)?|youtube)', msg, re.I
        )
        journal_context = re.findall(
            r'journal(s)?', context, re.I
        )
        journal_msg = re.findall(
            r'(journal|paper|article|essay|dissertation)(s)?', msg, re.I
        )

        if len(video_context) + len(video_msg) > 0:
            caller = getattr(self, 'youtube')
            name = 'Video'
        elif len(journal_context) + len(journal_msg) > 0:
            caller = getattr(self, 'journal')
            name = 'Journal'
        else:
            caller = getattr(self, 'book')
            name = 'Book'

        # 2. get the keyword and refinement from query
        keyword = query.get('keyword')
        refinements = list(filter(
            lambda x: x.lower() in ['video', 'book', 'journal'],
            set(query.get('refinements', []))
        ))
        # 3. if refinement's length if bigger than 1
        if len(refinements) > 1:
            # then use refinement
            self._caller = [caller, refinements, name]
        else:
            self._caller = [caller, [keyword], name]

        # 4. return authors, data and actions

    def start(self, refine=False):
        authors = []
        date = []
        debug(self._caller[1])
        self._caller[1] = list(filter(
            lambda x: x.lower() not in ['video', 'book', 'journal'],
            self._caller[1]
        ))
        debug(self._caller[1])
        elems = self._caller[0](' '.join(self._caller[1]), authors, date)

        for elem in elems:
            elem = list(filter(lambda x: x, elem))

        if refine:
            remove = len(self._caller[1][1:]) * 2
            if remove < 2:
                remove = 2
            elems = elems[:-remove]

        return authors, date, elems, self._caller[2]

    def book(self, keyword, authors, date):
        elems = []
        image = (
            'https://s3-ap-southeast-1.amazonaws.com/'
            'intralogue-natural-language-processing/chatbot-image/book.png'
        )
        book = build(
            self._GOOGLE_BOOK_API_SERVICE_NAME,
            self._GOOGLE_BOOK_API_VERSION,
            developerKey=self._DEVELOPER_KEY
        )
        res = book.volumes().list(source='public', q=keyword).execute()
        for item in res.get('items', []):
            info = item.get('volumeInfo')
            authors.extend(info.get('authors', []))
            date.append(info.get('publishDate'))
            elems.append({
                'title': info.get('title'),
                'subtitle': info.get('subtitle'),
                'image': info.get('imageLinks', {}).get('medium', image),
                'url': info.get('infoLink'),
                'uuid': 'book+' + str(item.get('id')),
                'summary': info.get('description'),
                'authors': info.get('authors'),
                'publishDate': info.get('publishDate')
            })
        return elems

    def journal(self, keyword, authors, date):
        elems = []
        image = (
            'https://s3-ap-southeast-1.amazonaws.com/'
            'intralogue-natural-language-processing/chatbot-image/journal.png'
        )
        params = dict(lang='en')
        # 1. try encode keyword
        #    try with ascii
        #    except unicode then try utf-8
        #    if nothing works then don't encode
        params['prop'] = urllib.parse.quote('pageimages|pageterms')
        try:
            params['keyword'] = keyword.encode('ascii')
        except UnicodeError:
            params['keyword'] = keyword.encode('utf-8')
        else:
            params['keyword'] = keyword
        # 2. request keyword to wikipedia
        url = (
            'https://{lang}.wikipedia.org/w/api.php?'
            'action=query&formatversion=2&generator=prefixsearch&'
            'gpssearch={keyword}&gpslimit=10&prop={prop}&piprop=thumbnail&'
            'pithumbsize=200&pilimit=10&wbptterms=description&format=json'
        ).format(**params)
        res = requests.get(url).json()
        # 3. for each item get:
        #       pageid
        #       title
        #       descirption inside terms
        #       parse with nornal url(https://{lang}.wikipedia.org/wiki/{title})
        if res.get('query', {}).get('pages'):
            for item in res.get('query', {}).get('pages'):
                params['title'] = item.get('title')
                elems.append({
                    'title': params['title'],
                    'subtitle': item.get('terms', {}).get('description', [])[0],
                    'image': item.get('thumbnail', {}).get('source', image),
                    'url': 'https://{lang}.wikipedia.org/wiki/{title}'.format(
                        **params
                    ),
                    'uuid': 'journal+' + str(item.get('pageid')),
                    'summary': item.get('terms', {}).get('description')
                })
        return elems

    def youtube(self, keyword, authors, date):
        elems = []
        image = (
            'https://s3-ap-southeast-1.amazonaws.com/'
            'intralogue-natural-language-processing/chatbot-image/video.png'
        )
        # 1. build the youtube api
        video = build(
            self._YOUTUBE_API_SERVICE_NAME,
            self._YOUTUBE_API_VERSION,
            developerKey=self._DEVELOPER_KEY
        )
        # 2. request youtube api
        #       q is keyword
        #       part is id,snippet
        #       type is video
        #       maxResults is 10
        res = video.search().list(
            q=keyword,
            part='id,snippet',
            type='video',
            maxResults=10
        ).execute()
        # 3. for each item
        #       parse url as https://youtu.be/{videoId}
        #       get only:
        #           [id][videoId]
        #           url
        #           [snippet][title]
        #           [snippte][description]
        #           [snippet][publishedAt]
        for item in res.get('items', []):
            if item['id']['kind'] == 'youtube#video':
                snippet = item.get('snippet', {})
                thumbnails = snippet.get('thumbnails')
                authors.append(snippet.get('channelTitle'))
                date.append(snippet.get('publishedAt'))
                elems.append({
                    'title': snippet.get('title'),
                    'subtitle': snippet.get('description'),
                    'image': thumbnails.get('medium', {}).get(
                        'url', image
                    ),
                    'url': 'https://youtu.be/{}'.format(
                        item.get('id', {}).get('videoId')
                    ),
                    'uuid': 'video+' + str(item.get('id', {}).get('videoId')),
                    'summary': snippet.get('description'),
                    'authors': snippet.get('channelTitle'),
                    'publishDate': snippet.get('publishedAt')
                })
        return elems
