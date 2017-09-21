#!/usr/bin/python
# -*- coding: utf-8 -*-
from apiclient.discovery import build
from apiclient.errors import HttpError
import requests
import urllib.parse

DEVELOPER_KEY = "AIzaSyAnlk2LuHMdmISsnJ6-5UFRpno685dwKgg"
GOOGLE_BOOK_API_SERVICE_NAME = "books"
GOOGLE_BOOK_API_VERSION = "v1"

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def book_search_request(keyword):
	googlebook = build(GOOGLE_BOOK_API_SERVICE_NAME, GOOGLE_BOOK_API_VERSION, developerKey=DEVELOPER_KEY)
	req = googlebook.volumes().list(source="public", q=keyword)
	response = req.execute()
	bookResult = []
	for book in response.get('items', []):
		# print(book);
		thumb = ""
		if "imageLinks" in book["volumeInfo"]:
			if "thumbnail" in book["volumeInfo"]["imageLinks"]:
				thumb = book["volumeInfo"]["imageLinks"]["thumbnail"]
		bookResult.append({
			"id": "book+" + book["id"],
			"domain": "https://books.google.com/",
			"authors": book["volumeInfo"]["authors"] if "authors" in book["volumeInfo"] else [],
			"thumbnail": thumb,
			"title": book["volumeInfo"]["title"] if "title" in book["volumeInfo"] else "",
			"subtitle": book["volumeInfo"]["subtitle"] if "subtitle" in book["volumeInfo"] else "",
			"description": book["volumeInfo"]["description"] if "description" in book["volumeInfo"] else "",
			"pageCount": book["volumeInfo"]["pageCount"] if "pageCount" in book["volumeInfo"] else "",
			"language": book["volumeInfo"]["language"] if "language" in book["volumeInfo"] else "",
			"publishedDate": book["volumeInfo"]["publishedDate"] if 'publishedDate' in book["volumeInfo"] else "",
			"url": book["volumeInfo"]["infoLink"]
		})
	return bookResult

def journal_search_request(keyword):
	lang = 'en'
	try:
		url = 'https://%s.wikipedia.org/w/api.php?action=query&formatversion=2&generator=prefixsearch&gpssearch=%s&gpslimit=10&prop=%s&piprop=thumbnail&pithumbsize=200&pilimit=10&wbptterms=description&format=json' % (lang, keyword.encode("ascii"), urllib.parse.quote('pageimages|pageterms'));
	except UnicodeError:
		url = 'https://%s.wikipedia.org/w/api.php?action=query&formatversion=2&generator=prefixsearch&gpssearch=%s&gpslimit=10&prop=%s&piprop=thumbnail&pithumbsize=200&pilimit=10&wbptterms=description&format=json' % (lang, keyword.encode("utf-8"), urllib.parse.quote('pageimages|pageterms'));
	else:
		url = 'https://%s.wikipedia.org/w/api.php?action=query&formatversion=2&generator=prefixsearch&gpssearch=%s&gpslimit=10&prop=%s&piprop=thumbnail&pithumbsize=200&pilimit=10&wbptterms=description&format=json' % (lang, keyword, urllib.parse.quote('pageimages|pageterms'));
	print(url)
	r = requests.get(url)
	response_json = r.json()
	actual_response = []
	print(response_json)
	if 'query' in response_json and 'pages' in response_json['query']:
		for item in response_json['query']['pages']:
			actual_response.append({
				"id": "journal+" + str(item['pageid']),
				"title": item["title"],
				"description": item["terms"]["description"][0] if 'terms' in item and 'description' in item['terms'] else "",
				"url": "https://en.wikipedia.org/wiki/%s" % item["title"]
				})
	return actual_response


def video_search_request(keyword):
	youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
	response = youtube.search().list(
		q=keyword,
		part="id,snippet",
		type="video",
		maxResults=10
		).execute()
	actual_response = []
	for video in response.get("items", []):
		if video["id"]["kind"] == "youtube#video":
			url = "https://www.youtube.com/watch?v=%s" % video["id"]["videoId"]
			actual_response.append({
				"id": "video+" + video["id"]["videoId"],
				"url": url,
				"title": video["snippet"]["title"],
				"description": video["snippet"]["description"],
				"publishedDate": video["snippet"]["publishedAt"]
				})
	return actual_response